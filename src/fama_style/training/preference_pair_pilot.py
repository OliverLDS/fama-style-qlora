"""Bounded grouped DPO versus SFT continuation; no ML imports during CPU checks."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time

try:
    from fama_style.evaluation import preference_token_preflight as tokens
except ModuleNotFoundError:
    import preference_token_preflight as tokens

PROTOCOL='actual_error_preference_pilot_v1'
MODEL_ID=tokens.MODEL_ID
REVISION=tokens.REVISION
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
HYPER={'epochs':2,'updates_per_arm':22,'learning_rate':1e-6,'beta':0.1,'seed':1729,'max_length':1024,
       'max_grad_norm':1.0,'weight_decay':0.0,'adam_beta1':0.9,'adam_beta2':0.999,'adam_epsilon':1e-8}
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
TARGET_PATTERN=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$'
ARMS=('sft','dpo')


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(source):return INSTRUCTION+'\n\nPassage:\n'+source
def messages(row):return [{'role':'user','content':[{'type':'text','text':row['prompt']}]}]


def schedule(rows,seed=1729):
    groups={}
    for r in rows:groups.setdefault(r['anchor_group_id'],[]).append(r['pair_id'])
    rng=random.Random(seed);result=[]
    for epoch in (1,2):
        order=sorted(groups);rng.shuffle(order)
        result.append({'epoch':epoch,'groups':[{'group_id':g,'pair_ids':groups[g]} for g in order]})
    return result


def preference_coefficients(pc,pr,rc,rr,beta=0.1):
    value=tokens.dpo_loss(pc,pr,rc,rr,beta)
    z=beta*((pc-pr)-(rc-rr))
    if z>=0:e=math.exp(-z);p=e/(1+e)
    else:e=math.exp(z);p=1/(1+e)
    coefficient=beta*p
    return {'loss':value,'chosen_gradient':-coefficient,'rejected_gradient':coefficient,'relative_margin':z/beta}


def validate_sequence(seq,limit=1024):
    ids=seq['input_ids'];labels=seq['labels']
    if (not ids or len(ids)!=len(labels) or len(ids)>limit or any(type(v) is not int or v<0 for v in ids)):
        raise ValueError('encoded_sequence_shape_failed')
    starts=[i for i,v in enumerate(labels) if v!=-100]
    if not starts or starts[0]<1 or labels!=[-100]*starts[0]+ids[starts[0]:]:raise ValueError('contiguous_response_mask_required')
    return starts[0],len(starts)


def load_inputs(data_path,config_path):
    config=json.loads(config_path.read_text())
    if (config['protocol']!=PROTOCOL or config['model_id']!=MODEL_ID or config['model_revision']!=REVISION
            or config['hyperparameters']!=HYPER or config['decoding']!=DECODING or config['training_authorized'] is not False
            or config['max_runtime_seconds']!=3600 or config['dataset_sha256']!=file_sha(data_path)
            or config['token_helper_sha256']!=file_sha(Path(tokens.__file__))
            or config['runtime_source_sha256']!=file_sha(Path(__file__))):raise ValueError('frozen_pilot_configuration_changed')
    data=json.loads(data_path.read_text());pairs=data['training_pairs'];development=data['development']
    if (len(pairs)!=17 or len({r['pair_id'] for r in pairs})!=17 or len({r['anchor_group_id'] for r in pairs})!=11
            or len(development)!=8 or len({r['example_id'] for r in development})!=8
            or [r['example_id'] for r in development]!=config['development_ids']):raise ValueError('frozen_train_and_eight_case_coverage_required')
    counts=Counter(r['anchor_group_id'] for r in pairs)
    for row in pairs:
        if (row['split']!='training' or row['anchor_balanced_weight']!=1/counts[row['anchor_group_id']]
                or row['prompt']!=prompt(row['source'])
                or any(sha(row[k])!=row[k+'_sha256'] for k in ('prompt','chosen','rejected'))):raise ValueError('training_text_or_weight_changed')
        prefixes=[]
        for kind in ('chosen','rejected'):
            seq=row['encoded'][kind];n,_=validate_sequence(seq);prefixes.append(seq['input_ids'][:n])
            expected=row['token_audits'][kind]
            if sha(json.dumps(seq['input_ids']))!=expected['input_ids_sha256'] or sha(json.dumps(seq['labels']))!=expected['labels_sha256']:
                raise ValueError('verified_training_token_payload_changed')
        if prefixes[0]!=prefixes[1]:raise ValueError('training_preference_prefix_changed')
    if ({r['paper_id'] for r in pairs}&{r['paper_id'] for r in development}
            or {r['example_id'] for r in pairs}&{r['example_id'] for r in development}):raise ValueError('training_development_overlap')
    for row in development:
        n,_=validate_sequence(row['encoded'])
        if (row['split']!='development' or row['prompt']!=prompt(row['source']) or n+512>1024
                or row['prefix_ids']!=row['encoded']['input_ids'][:n]
                or sha(json.dumps(row['prefix_ids']))!=row['prefix_ids_sha256']
                or any(sha(row[k])!=row[k+'_sha256'] for k in ('source','target','prompt'))):
            raise ValueError('development_text_prefix_or_context_changed')
    if schedule(pairs)!=config['group_schedule']:raise ValueError('group_order_changed')
    return pairs,development,config


def verify_adapter(adapter,config):
    if any(file_sha(adapter/name)!=digest for name,digest in config['adapter_sha256'].items()):
        raise ValueError('attach_original_expanded_epoch_two_adapter')
    cfg=json.loads((adapter/'adapter_config.json').read_text())
    if cfg['base_model_name_or_path']!=MODEL_ID or cfg['r']!=8 or cfg['lora_alpha']!=16 or cfg['lora_dropout']!=0.05:
        raise ValueError('initial_adapter_configuration_changed')


def run_training(pairs,development,config,adapter,output,max_seconds=3600):
    if output.exists():raise ValueError('fresh_pilot_output_required')
    verify_adapter(adapter,config);output.mkdir(parents=True);started=time.monotonic();cap=min(max_seconds,3600)
    state={'complete':False,'phase':'start','training_called':False,'optimizer_steps':0,'arm_steps':{a:0 for a in ARMS},'current_arm':None}
    report={'protocol':PROTOCOL,'complete':False,'config':config,'training_called':False,'snapshots':[],'arms':{},'reference_scores':[]}
    gradients=[];update_logs=[];forward_counts=Counter()
    def persist():write_json(output/'state.json',state);write_json(output/'evaluation.json',report)
    def budget():
        if time.monotonic()-started>=cap:raise TimeoutError('pilot_time_budget_exceeded')
    def phase(name):state['phase']=name;persist();print('Phase: '+name,flush=True);budget()
    persist()
    try:
        import torch
        import torch.nn.functional as F
        from huggingface_hub import HfApi
        from transformers import (AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,
                                  GenerationConfig,LogitsProcessor,LogitsProcessorList)
        from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose only GPU 0 on Kaggle T4 x2')
        packages={p:importlib.metadata.version(p) for p in config['packages']}
        if packages!=config['packages']:raise RuntimeError('Recorded control environment required')
        report['runtime']={'packages':packages,'model_id':MODEL_ID,'model_revision':REVISION,'gpu':torch.cuda.get_device_name(0),
            'gpu_count':1,'runtime_source_sha256':file_sha(Path(__file__)),'token_helper_sha256':file_sha(Path(tokens.__file__)),
            'implementation':'explicit grouped PyTorch loop; no TRL preprocessing or reference defaults'}
        torch.cuda.set_device(0);torch.manual_seed(HYPER['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('model_access')
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
        if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:raise RuntimeError('model_revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        phase('processor_equivalence');audits=[]
        def check_encoding(row,text,encoded):
            full=processor.apply_chat_template(messages(row)+[{'role':'assistant','content':[{'type':'text','text':text}]}],
                tokenize=True,return_dict=True,return_tensors='pt')
            ids=full['input_ids'][0].tolist()
            if ids!=encoded['input_ids'] or ('token_type_ids' in full and bool(full['token_type_ids'].any())):
                raise ValueError('processor_does_not_match_verified_text_only_ids')
            n,_=validate_sequence(encoded)
            prefix=processor.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
            if prefix!=ids[:n] or tokens.norm(tokenizer.decode(ids[n:],skip_special_tokens=True))!=tokens.norm(text):
                raise ValueError('processor_response_boundary_or_decode_changed')
            return {'full_tokens':len(ids),'prompt_tokens':n,'input_ids_sha256':sha(json.dumps(ids)),'labels_sha256':sha(json.dumps(encoded['labels']))}
        for row in pairs:
            for kind in ('chosen','rejected'):audits.append({'pair_id':row['pair_id'],'candidate':kind,**check_encoding(row,row[kind],row['encoded'][kind])})
        for row in development:audits.append({'example_id':row['example_id'],'candidate':'development_target',**check_encoding(row,row['target'],row['encoded'])})
        write_json(output/'processor_audit.json',audits)
        phase('load_model')
        base=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;base.config.use_cache=False;base.config.text_config.use_cache=False
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.0
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        matched=[n for n,_ in model.named_modules() if re.fullmatch(TARGET_PATTERN,n)]
        if (len(matched)!=2*model.config.text_config.num_hidden_layers or len(trainable)!=136
                or sum(p.numel() for p in trainable.values())!=2228224
                or any('lora_' not in n or 'language_model' not in n or p.dtype!=torch.float32 for n,p in trainable.items())):
            raise RuntimeError('only_original_language_qv_adapters_may_train')
        initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        def fingerprint(values):
            digest=hashlib.sha256()
            for name in sorted(values):
                value=values[name].detach().cpu().contiguous();digest.update(name.encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
            return digest.hexdigest()
        initial_hash=fingerprint(initial);report['initial_adapter_tensor_sha256']=initial_hash
        generation=GenerationConfig(**DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        if generation.to_dict()!=config['expected_generation_config']:raise ValueError('recorded_generation_configuration_changed')
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('nonfinite_generation_scores')
                return scores
        def inputs(seq):
            ids=torch.tensor([seq['input_ids']],dtype=torch.long,device='cuda:0')
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'token_type_ids':torch.zeros_like(ids)}
        def total_logp(seq,tag):
            budget();n,count=validate_sequence(seq);batch=inputs(seq)
            positions=torch.arange(n-1,len(seq['input_ids'])-1,device='cuda:0')
            outputs=model(**batch,use_cache=False,logits_to_keep=positions)
            logits=outputs.logits[0].float();targets=batch['input_ids'][0,n:]
            if logits.shape[0]!=count:raise RuntimeError('causal_selected_logits_shape_failed')
            value=-F.cross_entropy(logits,targets,reduction='sum')
            if not bool(torch.isfinite(value)):raise RuntimeError('nonfinite_response_log_probability')
            forward_counts[tag]+=1
            forward_counts[tag+'_input_tokens']+=len(seq['input_ids'])
            forward_counts[tag+'_response_tokens']+=count
            budget();return value,count
        def score(seq,tag):
            with torch.no_grad():v,n=total_logp(seq,tag)
            return {'total_log_probability':float(v.item()),'mean_nll':-float(v.item())/n,'supervised_tokens':n}
        def builtin_check(seq):
            manual=score(seq,'builtin_sanity_manual');batch=inputs(seq)
            with torch.no_grad():loss=float(model(**batch,labels=torch.tensor([seq['labels']],device='cuda:0'),use_cache=False).loss.item())
            forward_counts['builtin_sanity_loss']+=1
            if not math.isclose(loss,manual['mean_nll'],rel_tol=1e-5,abs_tol=1e-5):raise RuntimeError('manual_and_builtin_causal_loss_differ')
            return {'manual_mean_nll':manual['mean_nll'],'builtin_mean_nll':loss,'passed':True}
        model.eval();phase('likelihood_sanity')
        report['likelihood_sanity']=[builtin_check(pairs[0]['encoded']['chosen']),builtin_check(development[0]['encoded'])]
        def generate(row):
            budget();ids=torch.tensor([row['prefix_ids']],dtype=torch.long,device='cuda:0')
            with torch.inference_mode():result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=torch.zeros_like(ids),
                generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
            new=result[0,ids.shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True);eos=generation.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            result={'example_id':row['example_id'],'source_sha256':row['source_sha256'],'prompt_sha256':row['prompt_sha256'],
                'output':text,'output_tokens':int(new.numel()),'empty':not bool(text.strip()),
                'output_limited':bool(new.numel()>=512 and int(new[-1]) not in eos),'copied_source':tokens.norm(text)==tokens.norm(row['source'])}
            forward_counts['generation_requests']+=1;budget();return result
        def snapshot(arm,epoch,step):
            was_training=model.training;model.eval();phase('snapshot_'+arm+'_'+str(epoch))
            snap={'arm':arm,'epoch':epoch,'optimizer_step':step,'complete':False,'losses':[],'generations':[]};report['snapshots'].append(snap)
            for row in development:
                values=score(row['encoded'],'development_likelihood');snap['losses'].append({'example_id':row['example_id'],**values})
                g=generate(row);snap['generations'].append(g);persist();print(arm+' epoch '+str(epoch)+': saved %d/8 outputs'%len(snap['generations']),flush=True)
                if arm=='control' and any(g[k]!=row['expected_generation'][k] for k in ('output','output_tokens','empty','output_limited','copied_source')):
                    raise RuntimeError('unchanged_adapter_does_not_reproduce_archived_output')
            snap['token_weighted_development_nll']=-sum(r['total_log_probability'] for r in snap['losses'])/sum(r['supervised_tokens'] for r in snap['losses'])
            snap['complete']=True;persist();model.train(was_training)
        snapshot('control',0,0);report['common_control_exact_matches']=8
        phase('cache_initial_adapter_reference');reference={}
        for row in pairs:
            reference[row['pair_id']]={kind:score(row['encoded'][kind],'frozen_reference') for kind in ('chosen','rejected')}
            report['reference_scores'].append({'pair_id':row['pair_id'],**reference[row['pair_id']]});persist()
        if fingerprint(get_peft_model_state_dict(model))!=initial_hash:raise RuntimeError('reference_scoring_changed_adapter')
        cache_hash=sha(json.dumps(reference,sort_keys=True));write_json(output/'reference_scores.json',{
            'initial_adapter_tensor_sha256':initial_hash,'cache_sha256':cache_hash,'scores':reference,
            'dataset_sha256':config['dataset_sha256'],'model_id':MODEL_ID,'model_revision':REVISION,
            'token_helper_sha256':config['token_helper_sha256'],
            'reference_policy':'unchanged_expanded_epoch_2','source_adapter_sha256':config['adapter_sha256']})
        by_id={r['pair_id']:r for r in pairs}
        for arm in ARMS:
            arm_started=time.monotonic()
            phase('initialize_'+arm);state['current_arm']=arm;torch.manual_seed(HYPER['seed'])
            for p in trainable.values():p.grad=None
            set_peft_model_state_dict(model,initial)
            if fingerprint(get_peft_model_state_dict(model))!=initial_hash:raise RuntimeError('independent_starting_adapter_reset_failed')
            optimizer=torch.optim.AdamW(list(trainable.values()),lr=HYPER['learning_rate'],betas=(0.9,0.999),eps=1e-8,weight_decay=0.)
            if optimizer.state:raise RuntimeError('optimizer_must_start_fresh')
            model.train();identity=[]
            for row in pairs:
                current={k:score(row['encoded'][k],'initial_policy_identity') for k in ('chosen','rejected')}
                deltas=[abs(current[k]['total_log_probability']-reference[row['pair_id']][k]['total_log_probability']) for k in current]
                loss=tokens.dpo_loss(*(current[k]['total_log_probability'] for k in ('chosen','rejected')),
                    *(reference[row['pair_id']][k]['total_log_probability'] for k in ('chosen','rejected')),HYPER['beta'])
                if max(deltas)>1e-3 or abs(loss-math.log(2))>2e-4:raise RuntimeError('initial_policy_reference_identity_failed')
                identity.append({'pair_id':row['pair_id'],'max_logp_difference':max(deltas),'dpo_loss':loss})
            report['arms'][arm]={'initial_adapter_tensor_sha256':initial_hash,'reference_identity':identity,'optimizer_initially_empty':True}
            for epoch in config['group_schedule']:
                phase(arm+'_epoch_'+str(epoch['epoch']))
                for group in epoch['groups']:
                    budget();optimizer.zero_grad(set_to_none=True);state['current_group_id']=group['group_id'];persist()
                    state['training_called']=True;report['training_called']=True;persist()
                    losses=[];count=len(group['pair_ids'])
                    for pid in group['pair_ids']:
                        row=by_id[pid]
                        if arm=='sft':
                            value,n=total_logp(row['encoded']['chosen'],'sft_backward');loss=-value/n/count
                            losses.append(float((-value/n).detach().item()));loss.backward();del value,loss
                        else:
                            policy={k:score(row['encoded'][k],'dpo_policy_nograd') for k in ('chosen','rejected')}
                            ref=reference[pid];coeff=preference_coefficients(policy['chosen']['total_log_probability'],policy['rejected']['total_log_probability'],
                                ref['chosen']['total_log_probability'],ref['rejected']['total_log_probability'],HYPER['beta']);losses.append(coeff['loss'])
                            for kind in ('chosen','rejected'):
                                value,_=total_logp(row['encoded'][kind],'dpo_backward')
                                if not math.isclose(float(value.detach().item()),policy[kind]['total_log_probability'],rel_tol=1e-6,abs_tol=1e-3):
                                    raise RuntimeError('two_pass_policy_logp_not_deterministic')
                                (value*(coeff[kind+'_gradient']/count)).backward();del value
                    grads=[p.grad for p in trainable.values() if p.grad is not None]
                    if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('nonfinite_adapter_gradients')
                    grad_norm=math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
                    if not math.isfinite(grad_norm) or grad_norm<=0:raise RuntimeError('nonpositive_adapter_gradient')
                    step=state['arm_steps'][arm]+1;lr=HYPER['learning_rate']*(1-(step-1)/22)
                    for pg in optimizer.param_groups:pg['lr']=lr
                    torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True)
                    if sha(json.dumps(reference,sort_keys=True))!=cache_hash:raise RuntimeError('frozen_reference_cache_mutated')
                    state['training_called']=True;report['training_called']=True;optimizer.step()
                    state['arm_steps'][arm]=step;state['optimizer_steps']+=1
                    gradients.append({'arm':arm,'step':step,'finite':True,'norm':grad_norm})
                    update_logs.append({'arm':arm,'epoch':epoch['epoch'],'step':step,'group_id':group['group_id'],'rows':count,
                        'loss':sum(losses)/count,'learning_rate':lr,'reference_cache_sha256':cache_hash})
                    write_json(output/'gradient_checks.json',gradients);write_json(output/'update_log.json',update_logs);persist()
                    print(arm+': completed %d/22 updates'%step,flush=True)
                if state['arm_steps'][arm]!=11*epoch['epoch']:raise RuntimeError('group_epoch_boundary_changed')
                folder=output/arm/('epoch_'+str(epoch['epoch']))/'adapter';model.save_pretrained(str(folder))
                snapshot(arm,epoch['epoch'],state['arm_steps'][arm]);model.train()
            final=fingerprint(get_peft_model_state_dict(model))
            if final==initial_hash:raise RuntimeError('adapter_weights_did_not_change')
            model.eval();final_scores=[]
            for row in pairs:final_scores.append({'pair_id':row['pair_id'],**{k:score(row['encoded'][k],'final_training_likelihood') for k in ('chosen','rejected')}})
            before_reload={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            del optimizer
            unwrapped=model.unload();model=PeftModel.from_pretrained(unwrapped,str(output/arm/'epoch_2/adapter'),is_trainable=True)
            reloaded=get_peft_model_state_dict(model)
            if set(reloaded)!=set(before_reload) or any(not torch.equal(v,reloaded[k].detach().cpu()) for k,v in before_reload.items()):
                raise RuntimeError('saved_adapter_reload_mismatch')
            for module in model.modules():
                if isinstance(module,torch.nn.Dropout):module.p=0.
            trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
            report['arms'][arm].update(final_adapter_tensor_sha256=final,reload_weights_match=True,weights_changed=True,
                elapsed_seconds=time.monotonic()-arm_started,
                completed_updates=state['arm_steps'][arm],final_training_scores=final_scores,
                final_adapter_files_sha256={name:file_sha(output/arm/'epoch_2/adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')})
            persist()
        if state['optimizer_steps']!=44 or len(gradients)!=44 or len(report['snapshots'])!=5:raise RuntimeError('bounded_coverage_incomplete')
        report.update(complete=True,forward_counts=dict(forward_counts),frozen_reference_cache_sha256=cache_hash)
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Bounded pilot stopped; preserve state and partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--train-authorized',action='store_true')
    p.add_argument('--max-runtime-seconds',type=int,default=3600)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/actual_error_pilot_v1'))
    a=p.parse_args(argv);pairs,development,config=load_inputs(a.data,a.config)
    if a.adapter is not None:verify_adapter(a.adapter,config)
    if a.check:print('Validated 17 training pairs / 11 groups / 8 development cases; two independent 22-update arms. No model or training.');return
    if not a.train_authorized or a.adapter is None or not 0<a.max_runtime_seconds<=3600:raise ValueError('explicit_bounded_training_authorization_required')
    run_training(pairs,development,config,a.adapter,a.output,a.max_runtime_seconds)


if __name__=='__main__':main()
