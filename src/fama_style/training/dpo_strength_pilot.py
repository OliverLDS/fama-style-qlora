"""Matched bounded DPO continuation; fixed starting-adapter reference and beta 0.1."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import tempfile
import time

try:
    from fama_style.training import preference_pair_pilot as prior
except ModuleNotFoundError:
    import preference_pair_pilot as prior

PROTOCOL='dpo_strength_pilot_v1'
HYPER=dict(prior.HYPER)
HYPER['learning_rate']=1e-5


def load_inputs(data_path,config_path):
    config=json.loads(config_path.read_text());payload=json.loads(data_path.read_text())
    if (config['protocol']!=PROTOCOL or config['hyperparameters']!=HYPER or config['training_authorized'] is not True
            or config['max_runtime_seconds']!=1800 or config['dataset_sha256']!=prior.file_sha(data_path)
            or config['runtime_source_sha256']!=prior.file_sha(Path(__file__))
            or config['prior_runtime_sha256']!=prior.file_sha(Path(prior.__file__))
            or config['token_helper_sha256']!=prior.file_sha(Path(prior.tokens.__file__))
            or prior.sha(payload['pilot_data_json'])!=config['prior_data_sha256']
            or prior.sha(payload['pilot_config_json'])!=config['prior_config_sha256']):raise ValueError('frozen_strength_pilot_changed')
    # Reuse the old CPU validator with byte-identical original artifacts, never alter its globals.
    with tempfile.TemporaryDirectory(prefix='fama-sft-contract-') as temp:
        folder=Path(temp);dp=folder/'data.json';cp=folder/'config.json'
        dp.write_text(payload['pilot_data_json']);cp.write_text(payload['pilot_config_json'])
        pairs,development,previous=prior.load_inputs(dp,cp)
    if (previous['model_id']!=config['model_id'] or previous['model_revision']!=config['model_revision']
            or previous['adapter_sha256']!=config['adapter_sha256'] or previous['packages']!=config['packages']
            or previous['expected_generation_config']!=config['expected_generation_config']
            or previous['decoding']!=config['decoding'] or previous['group_schedule']!=config['group_schedule']):
        raise ValueError('starting_adapter_environment_or_group_order_changed')
    cases=payload['training_cases'];by_id={p['pair_id']:p for p in pairs}
    if set(config['initial_chosen_scores'])!=set(by_id) or set(config['reference_scores'])!=set(by_id):
        raise ValueError('seventeen_frozen_reference_pairs_required')
    for pid,row in by_id.items():
        for kind in ('chosen','rejected'):
            score=config['reference_scores'][pid][kind];n=prior.validate_sequence(row['encoded'][kind])[1]
            if (score['supervised_tokens']!=n or not math.isfinite(score['total_log_probability'])
                    or score['total_log_probability']>0 or not math.isclose(score['mean_nll'],-score['total_log_probability']/n)):
                raise ValueError('invalid_frozen_reference_score')
        if config['initial_chosen_scores'][pid]!=config['reference_scores'][pid]['chosen']:
            raise ValueError('chosen_reference_score_mismatch')
    if (len(cases)!=11 or [r['pair_id'] for r in cases]!=config['training_case_ids']
            or len({r['anchor_group_id'] for r in cases})!=11):raise ValueError('eleven_frozen_training_cases_required')
    for row in cases:
        original=by_id[row['pair_id']];seq=original['encoded']['chosen'];n,_=prior.validate_sequence(seq)
        if (any(row[k]!=original[k] for k in ('source','prompt','source_sha256','prompt_sha256','anchor_group_id','example_id','paper_id','input_kind'))
                or row['prefix_ids']!=seq['input_ids'][:n]
                or prior.sha(json.dumps(row['prefix_ids']))!=row['prefix_ids_sha256'] or n+512>1024):
            raise ValueError('training_generation_input_changed')
    return pairs,development,cases,config


def learning_rate(step):
    if type(step) is not int or not 1<=step<=22:raise ValueError('step_outside_fixed_budget')
    return HYPER['learning_rate']*(1-(step-1)/22)


def run_training(pairs,development,cases,config,adapter,output,max_seconds=1800):
    if output.exists():raise ValueError('fresh_strength_pilot_output_required')
    prior.verify_adapter(adapter,config);output.mkdir(parents=True);started=time.monotonic()
    state={'complete':False,'phase':'start','training_called':False,'optimizer_steps':0}
    report={'protocol':PROTOCOL,'complete':False,'config':config,'training_called':False,
        'control_generations':[],'initial_training_scores':[],'final_training_scores':[],
        'initial_development_scores':[],'final_development_scores':[],'final_generations':[],'checkpoints':[]}
    counts=Counter();gradients=[];updates=[]
    def persist():prior.write_json(output/'state.json',state);prior.write_json(output/'evaluation.json',report)
    def budget():
        if time.monotonic()-started>=min(max_seconds,1800):raise TimeoutError('strength_pilot_time_budget_exceeded')
    def phase(name):state['phase']=name;persist();print('Phase: '+name,flush=True);budget()
    persist()
    try:
        import torch
        import torch.nn.functional as F
        from huggingface_hub import HfApi
        from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,GenerationConfig,LogitsProcessor,LogitsProcessorList
        from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose only GPU 0 on Kaggle T4 x2')
        packages={p:importlib.metadata.version(p) for p in config['packages']}
        if packages!=config['packages']:raise RuntimeError('recorded_environment_required')
        report['runtime']={'packages':packages,'gpu':torch.cuda.get_device_name(0),'gpu_count':1,
            'model_id':config['model_id'],'model_revision':config['model_revision'],
            'runtime_source_sha256':prior.file_sha(Path(__file__)),'prior_runtime_sha256':prior.file_sha(Path(prior.__file__)),
            'token_helper_sha256':prior.file_sha(Path(prior.tokens.__file__))}
        torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats(0);phase('model_access')
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
        if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
        if HfApi().model_info(prior.MODEL_ID,revision=prior.REVISION,token=token,timeout=60).sha!=prior.REVISION:
            raise RuntimeError('model_revision_changed')
        processor=AutoProcessor.from_pretrained(prior.MODEL_ID,revision=prior.REVISION,token=token);tokenizer=processor.tokenizer
        phase('processor_equivalence');audits=[]
        for row,text,seq,identifier in ([(r,r[kind],r['encoded'][kind],r['pair_id']+'__'+kind) for r in pairs for kind in ('chosen','rejected')]+
                                      [(r,r['target'],r['encoded'],r['example_id']) for r in development]):
            full=processor.apply_chat_template(prior.messages(row)+[{'role':'assistant','content':[{'type':'text','text':text}]}],
                tokenize=True,return_dict=True,return_tensors='pt')
            prefix=processor.apply_chat_template(prior.messages(row),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
            ids=full['input_ids'][0].tolist();n,_=prior.validate_sequence(seq)
            if (ids!=seq['input_ids'] or prefix!=ids[:n] or ('token_type_ids' in full and bool(full['token_type_ids'].any()))
                    or prior.tokens.norm(tokenizer.decode(ids[n:],skip_special_tokens=True))!=prior.tokens.norm(text)):
                raise ValueError('verified_response_sequence_or_boundary_changed')
            audits.append({'id':identifier,'input_ids_sha256':prior.sha(json.dumps(ids)),
                'labels_sha256':prior.sha(json.dumps(seq['labels'])),'prompt_tokens':n,'full_tokens':len(ids)})
        prior.write_json(output/'processor_audit.json',audits)
        phase('load_model')
        base=Gemma3ForConditionalGeneration.from_pretrained(prior.MODEL_ID,revision=prior.REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;base.config.use_cache=False;base.config.text_config.use_cache=False
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        if (len(trainable)!=136 or sum(p.numel() for p in trainable.values())!=2228224
                or len([n for n,_ in model.named_modules() if re.fullmatch(prior.TARGET_PATTERN,n)])!=2*model.config.text_config.num_hidden_layers
                or any('lora_' not in n or 'language_model' not in n or p.dtype!=torch.float32 for n,p in trainable.items())):
            raise RuntimeError('only_original_language_qv_adapters_may_train')
        def fingerprint(values):
            digest=hashlib.sha256()
            for name in sorted(values):
                value=values[name].detach().cpu().contiguous();digest.update(name.encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
            return digest.hexdigest()
        initial=fingerprint(get_peft_model_state_dict(model));report['initial_adapter_tensor_sha256']=initial
        if initial!=config['initial_adapter_tensor_sha256']:raise RuntimeError('starting_adapter_tensor_changed')
        generation=GenerationConfig(**prior.DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        if generation.to_dict()!=config['expected_generation_config']:raise ValueError('decoding_changed')
        prior.write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('nonfinite_generation_scores')
                return scores
        def inputs(seq):
            ids=torch.tensor([seq['input_ids']],device='cuda:0',dtype=torch.long)
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'token_type_ids':torch.zeros_like(ids)}
        def total_logp(seq,tag):
            budget();n,count=prior.validate_sequence(seq);batch=inputs(seq)
            output=model(**batch,use_cache=False,logits_to_keep=torch.arange(n-1,len(seq['input_ids'])-1,device='cuda:0'))
            value=-F.cross_entropy(output.logits[0].float(),batch['input_ids'][0,n:],reduction='sum')
            if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite_likelihood')
            counts[tag]+=1;counts[tag+'_input_tokens']+=len(seq['input_ids']);counts[tag+'_response_tokens']+=count
            return value,count
        def score(seq,tag):
            with torch.no_grad():value,n=total_logp(seq,tag)
            return {'total_log_probability':float(value.item()),'mean_nll':-float(value.item())/n,'supervised_tokens':n}
        def generate(row,split):
            budget();ids=torch.tensor([row['prefix_ids']],device='cuda:0',dtype=torch.long)
            with torch.inference_mode():result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=torch.zeros_like(ids),
                generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
            new=result[0,ids.shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True);eos=generation.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            g={'split':split,'example_id':row['example_id'],'pair_id':row.get('pair_id'),'source_sha256':row['source_sha256'],
                'prompt_sha256':row['prompt_sha256'],'output':text,'output_sha256':prior.sha(text),'output_tokens':int(new.numel()),
                'empty':not bool(text.strip()),'output_limited':bool(new.numel()>=512 and int(new[-1]) not in eos),
                'copied_source':prior.tokens.norm(text)==prior.tokens.norm(row['source'])}
            counts['generation_requests']+=1;budget();return g
        model.eval();phase('likelihood_and_control_gates');sanity=[]
        for seq in (pairs[0]['encoded']['chosen'],pairs[0]['encoded']['rejected'],development[0]['encoded']):
            manual=score(seq,'sanity_manual')
            with torch.no_grad():builtin=float(model(**inputs(seq),labels=torch.tensor([seq['labels']],device='cuda:0'),use_cache=False).loss.item())
            if not math.isclose(builtin,manual['mean_nll'],rel_tol=1e-5,abs_tol=1e-5):raise RuntimeError('manual_builtin_likelihood_mismatch')
            sanity.append({'manual_mean_nll':manual['mean_nll'],'builtin_mean_nll':builtin,'passed':True})
        report['likelihood_sanity']=sanity
        for row in pairs:
            values=score(row['encoded']['chosen'],'initial_training_score');gold=config['initial_chosen_scores'][row['pair_id']]
            if values['supervised_tokens']!=gold['supervised_tokens'] or abs(values['total_log_probability']-gold['total_log_probability'])>1e-3:
                raise RuntimeError('starting_chosen_likelihood_changed')
            report['initial_training_scores'].append({'pair_id':row['pair_id'],**values})
        for row in development:
            report['initial_development_scores'].append({'example_id':row['example_id'],**score(row['encoded'],'initial_development_score')})
            g=generate(row,'development');report['control_generations'].append(g);persist()
            if any(g[k]!=row['expected_generation'][k] for k in row['expected_generation']):raise RuntimeError('eight_archived_controls_must_reproduce')
            print('Verified %d/8 unchanged-adapter controls'%len(report['control_generations']),flush=True)
        report['control_exact_matches']=8
        phase('verify_fixed_reference');reference={}
        initial_chosen={r['pair_id']:{k:r[k] for k in ('total_log_probability','mean_nll','supervised_tokens')}
            for r in report['initial_training_scores']}
        for row in pairs:
            pid=row['pair_id'];reference[pid]={'chosen':initial_chosen[pid],
                'rejected':score(row['encoded']['rejected'],'initial_rejected_reference')}
            for kind in ('chosen','rejected'):
                measured=reference[pid][kind];gold=config['reference_scores'][pid][kind]
                if measured['supervised_tokens']!=gold['supervised_tokens'] or abs(measured['total_log_probability']-gold['total_log_probability'])>1e-3:
                    raise RuntimeError('initial_reference_likelihood_changed')
        if fingerprint(get_peft_model_state_dict(model))!=initial:raise RuntimeError('reference_scoring_changed_adapter')
        cache_hash=prior.sha(json.dumps(reference,sort_keys=True))
        report['reference_cache_sha256']=cache_hash
        prior.write_json(output/'reference_scores.json',{'scores':reference,'cache_sha256':cache_hash,
            'initial_adapter_tensor_sha256':initial,'source_adapter_sha256':config['adapter_sha256'],
            'dataset_sha256':config['dataset_sha256'],'model_id':config['model_id'],'model_revision':config['model_revision'],
            'reference_policy':'unchanged_original_expanded_epoch_2'})
        phase('initial_policy_reference_identity');torch.manual_seed(1729);model.train();optimizer=torch.optim.AdamW(list(trainable.values()),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        report['optimizer_initially_empty']=not bool(optimizer.state);by_id={r['pair_id']:r for r in pairs}
        if optimizer.state:raise RuntimeError('fresh_optimizer_required')
        identity=[]
        for row in pairs:
            pid=row['pair_id'];current={k:score(row['encoded'][k],'initial_policy_identity') for k in ('chosen','rejected')}
            pc,pr=(current[k]['total_log_probability'] for k in ('chosen','rejected'))
            rc,rr=(reference[pid][k]['total_log_probability'] for k in ('chosen','rejected'))
            delta=max(abs(pc-rc),abs(pr-rr));loss=prior.tokens.dpo_loss(pc,pr,rc,rr,HYPER['beta'])
            if delta>1e-3 or abs(loss-math.log(2))>2e-4:raise RuntimeError('initial_policy_reference_identity_failed')
            identity.append({'pair_id':pid,'max_logp_difference':delta,'dpo_loss':loss})
        report['initial_policy_reference_identity']=identity;phase('train_dpo')
        for epoch in config['group_schedule']:
            for group in epoch['groups']:
                budget();optimizer.zero_grad(set_to_none=True);state['training_called']=True;report['training_called']=True;state['current_group_id']=group['group_id'];persist()
                losses=[]
                for pid in group['pair_ids']:
                    row=by_id[pid]
                    policy={k:score(row['encoded'][k],'dpo_policy_nograd') for k in ('chosen','rejected')}
                    pc,pr=(policy[k]['total_log_probability'] for k in ('chosen','rejected'))
                    rc,rr=(reference[pid][k]['total_log_probability'] for k in ('chosen','rejected'))
                    coefficient=prior.preference_coefficients(pc,pr,rc,rr,HYPER['beta']);losses.append(coefficient['loss'])
                    for kind in ('chosen','rejected'):
                        value,_=total_logp(row['encoded'][kind],'dpo_backward')
                        if not math.isclose(float(value.detach().item()),policy[kind]['total_log_probability'],rel_tol=1e-6,abs_tol=1e-3):
                            raise RuntimeError('two_pass_policy_not_deterministic')
                        (value*(coefficient[kind+'_gradient']/len(group['pair_ids']))).backward();del value
                grads=[p.grad for p in trainable.values() if p.grad is not None]
                if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('nonfinite_gradient')
                norm=math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
                if not math.isfinite(norm) or norm<=0:raise RuntimeError('nonpositive_gradient')
                step=state['optimizer_steps']+1;lr=learning_rate(step)
                for pg in optimizer.param_groups:pg['lr']=lr
                torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True)
                if prior.sha(json.dumps(reference,sort_keys=True))!=cache_hash:raise RuntimeError('reference_cache_mutated')
                optimizer.step();state['optimizer_steps']=step
                gradients.append({'step':step,'finite':True,'norm':norm});updates.append({'step':step,'epoch':epoch['epoch'],
                    'group_id':group['group_id'],'pair_ids':group['pair_ids'],'learning_rate':lr,'loss':sum(losses)/len(losses),'reference_cache_sha256':cache_hash})
                prior.write_json(output/'gradient_checks.json',gradients);prior.write_json(output/'update_log.json',updates);persist()
                print('Matched DPO: completed %d/22 updates'%step,flush=True)
            if state['optimizer_steps']!=11*epoch['epoch']:raise RuntimeError('epoch_boundary_changed')
            folder=output/('epoch_'+str(epoch['epoch']))/'adapter'
            model.save_pretrained(str(folder),save_embedding_layers=False)
            report['checkpoints'].append({'epoch':epoch['epoch'],'step':state['optimizer_steps'],'files_sha256':{
                name:prior.file_sha(folder/name) for name in ('adapter_config.json','adapter_model.safetensors')}});persist()
        phase('verify_reload');final=fingerprint(get_peft_model_state_dict(model))
        if final==initial:raise RuntimeError('weights_did_not_change')
        saved={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()};del optimizer
        model=PeftModel.from_pretrained(model.unload(),str(output/'epoch_2/adapter'),is_trainable=False)
        loaded=get_peft_model_state_dict(model)
        if set(loaded)!=set(saved) or any(not torch.equal(v,loaded[k].detach().cpu()) for k,v in saved.items()):raise RuntimeError('adapter_reload_mismatch')
        model.eval();model.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):raise RuntimeError('final_evaluation_must_be_frozen')
        report.update(weights_changed=True,adapter_reload_weights_match=True,final_adapter_tensor_sha256=final,final_evaluation_frozen=True)
        phase('final_evaluation')
        for row in pairs:report['final_training_scores'].append({'pair_id':row['pair_id'],**score(row['encoded']['chosen'],'final_training_score')})
        for row in development:report['final_development_scores'].append({'example_id':row['example_id'],**score(row['encoded'],'final_development_score')})
        for split,rows in (('training',cases),('development',development)):
            for row in rows:
                report['final_generations'].append(generate(row,split));persist();print('Saved %d/19 final generations'%len(report['final_generations']),flush=True)
        report['final_preference_scores']=[]
        chosen_by={r['pair_id']:{k:r[k] for k in ('total_log_probability','mean_nll','supervised_tokens')} for r in report['final_training_scores']}
        for row in pairs:
            report['final_preference_scores'].append({'pair_id':row['pair_id'],'chosen':chosen_by[row['pair_id']],
                'rejected':score(row['encoded']['rejected'],'final_rejected_score')})
        if prior.sha(json.dumps(reference,sort_keys=True))!=cache_hash:raise RuntimeError('reference_cache_mutated')
        if state['optimizer_steps']!=22 or len(report['final_generations'])!=19 or counts['generation_requests']!=27:
            raise RuntimeError('fixed_budget_or_generation_coverage_failed')
        report.update(complete=True,forward_counts=dict(counts));state.update(complete=True,phase='complete',
            elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Matched DPO stopped; preserve partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','config'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--train-authorized',action='store_true')
    p.add_argument('--max-runtime-seconds',type=int,default=1800);p.add_argument('--output',type=Path,default=Path('/kaggle/working/dpo_strength_v1'))
    a=p.parse_args(argv);pairs,dev,cases,config=load_inputs(a.data,a.config)
    if a.adapter is not None:prior.verify_adapter(a.adapter,config)
    if a.check:print('Validated identical 17 pairs / 22 DPO updates at 1e-5 / beta 0.1; 11 training + 8 development cases. No model or training.');return
    if not a.train_authorized or a.adapter is None or not 0<a.max_runtime_seconds<=1800:raise ValueError('explicit_bounded_dpo_authorization_required')
    run_training(pairs,dev,cases,config,a.adapter,a.output,a.max_runtime_seconds)


if __name__=='__main__':main()
