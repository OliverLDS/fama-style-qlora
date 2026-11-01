"""Small, isolated teacher-pair 4B baseline and QLoRA pilot; CPU checks are weight-free."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

PROTOCOL='teacher_pair_reverse_pilot_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
TARGET_PATTERN=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$'
HYPER={'epochs':2,'learning_rate':2e-5,'rank':8,'alpha':16,'dropout':0.05,
       'batch_size':1,'gradient_accumulation':2,'optimizer_steps':16,'max_length':1024,'seed':1729}
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(source):return INSTRUCTION+'\n\nPassage:\n'+source

def load_inputs(data_path,config_path):
    config=json.loads(config_path.read_text())
    if config['protocol']!=PROTOCOL or config['model_id']!=MODEL_ID or config['model_revision']!=REVISION:
        raise ValueError('model_or_protocol_changed')
    if config['hyperparameters']!=HYPER or config['decoding']!=DECODING:raise ValueError('experiment_settings_changed')
    if config['quantization']!='NF4 double quantization; FP32 compute':raise ValueError('precision_changed')
    if sha(data_path.read_text())!=config['dataset_sha256']:raise ValueError('dataset_changed')
    rows=[json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    groups={'training':[],'development':[]};seen=set();texts=set()
    fields={'example_id','paper_id','author_group_id','authors','split','source','target','source_sha256','target_sha256','prompt','pilot_eligible'}
    for row in rows:
        if set(row)!=fields or row['split'] not in groups or row['pilot_eligible'] is not True:raise ValueError('unapproved_row')
        if row['example_id'] in seen:raise ValueError('duplicate_example')
        seen.add(row['example_id'])
        if row['prompt']!=prompt(row['source']) or row['source_sha256']!=sha(row['source']) or row['target_sha256']!=sha(row['target']):
            raise ValueError('text_or_prompt_changed')
        for text in (row['source'],row['target']):
            normalized=' '.join(text.casefold().split())
            if not normalized or normalized in texts:raise ValueError('duplicate_or_empty_source_or_target')
            texts.add(normalized)
        groups[row['split']].append(row)
    if [len(groups[k]) for k in groups]!=[16,8]:raise ValueError('expected_16_training_8_development')
    for split,key in [('training','training_ids'),('development','development_ids')]:
        if [r['example_id'] for r in groups[split]]!=config[key]:raise ValueError('frozen_split_ids_changed')
    papers=[{r['paper_id'] for r in groups[k]} for k in groups]
    authors=[{a.strip().casefold() for r in groups[k] for a in r['authors']} for k in groups]
    if papers[0]&papers[1] or authors[0]&authors[1]:raise ValueError('paper_or_author_leakage')
    probes=config['training_probe_ids']
    if len(probes)!=2 or len(set(probes))!=2 or not set(probes)<={r['example_id'] for r in groups['training']}:
        raise ValueError('two_training_probes_required')
    return groups,config


def mask_labels(prefix,ids,decoded_target,target):
    if ids[:len(prefix)]!=prefix or len(ids)<=len(prefix) or len(ids)>HYPER['max_length']:
        raise ValueError('token_boundary_or_length_no_truncation')
    if ' '.join(decoded_target.split())!=' '.join(target.split()):raise ValueError('supervised_target_mismatch')
    return [-100]*len(prefix)+ids[len(prefix):]


def validate_baseline(path,config):
    report=json.loads(path.read_text())
    if report.get('complete') is not True or report.get('mode')!='baseline' or report.get('config')!=config:
        raise ValueError('attach_complete_baseline_for_same_frozen_dataset')
    snap=report['snapshots'][0]
    if (snap.get('complete') is not True or snap.get('name')!='base' or snap.get('optimizer_step')!=0
            or [r['example_id'] for r in snap['losses']]!=config['development_ids']
            or [r['example_id'] for r in snap['generations']]!=config['development_ids']+config['training_probe_ids']
            or not math.isfinite(snap['token_weighted_development_loss'])):
        raise ValueError('baseline_coverage_or_loss')
    return report


def load_hf_token():
    if Path('/kaggle').exists():
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
    else:token=os.environ.get('HF_TOKEN')
    if not isinstance(token,str) or not token.strip():raise RuntimeError('Configure HF_TOKEN outside the repository')
    return token


def run_training(groups,config,mode,output,baseline=None):
    """ML imports and all device/model execution remain inside this GPU entry point."""
    if output.exists() and any(output.iterdir()):raise ValueError('fresh_output_directory_required')
    if mode=='pilot' and baseline is None:raise ValueError('baseline_required_before_training')
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    state={'protocol':PROTOCOL,'mode':mode,'complete':False,'phase':'start','training_called':False,'optimizer_steps':0}
    report={'protocol':PROTOCOL,'mode':mode,'complete':False,'config':config,'snapshots':[],'training_called':False}
    def persist():
        write_json(output/'state.json',state);write_json(output/'evaluation.json',report)
    def phase(name):
        state['phase']=name;persist();print('Phase: '+name,flush=True)
    persist()
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import (AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,
            GenerationConfig,LogitsProcessor,LogitsProcessorList,Trainer,TrainerCallback,TrainingArguments)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose GPU 0 only on Kaggle T4 x2 before importing torch')
        torch.cuda.set_device(0);torch.manual_seed(HYPER['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('model_access');token=load_hf_token()
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:raise RuntimeError('revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        encoded={};audits=[]
        def messages(row):return [{'role':'user','content':[{'type':'text','text':row['prompt']}]}]
        for row in groups['training']+groups['development']:
            user=messages(row)
            prefix=processor.apply_chat_template(user,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
            full=processor.apply_chat_template(user+[{'role':'assistant','content':[{'type':'text','text':row['target']}]}],tokenize=True,return_dict=True,return_tensors='pt')
            ids=full['input_ids'][0].tolist()
            labels=mask_labels(prefix,ids,tokenizer.decode(ids[len(prefix):],skip_special_tokens=True),row['target'])
            if len(prefix)+DECODING['max_new_tokens']>HYPER['max_length']:raise ValueError('generation_context_budget')
            encoded[row['example_id']]={k:v[0].tolist() for k,v in full.items() if k in ('input_ids','attention_mask','token_type_ids')}
            encoded[row['example_id']]['labels']=labels
            audits.append({'example_id':row['example_id'],'split':row['split'],'full_tokens':len(ids),
                'masked_tokens':len(prefix),'supervised_tokens':len(ids)-len(prefix),'labels_match_target':True,
                'input_ids_sha256':sha(json.dumps(ids))})
        write_json(output/'label_audit.json',audits)
        packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','bitsandbytes','huggingface_hub')}
        report['runtime']={'model_id':MODEL_ID,'model_revision':REVISION,'packages':packages,
            'gpu':torch.cuda.get_device_name(0),'gpu_count':1,'runtime_source_sha256':sha(Path(__file__).read_text())}
        phase('load_model')
        model=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;model.eval();model.requires_grad_(False)
        generation=GenerationConfig(**DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('nonfinite_scores')
                return scores
        def batch(row):return {k:torch.tensor([v],dtype=torch.long,device='cuda:0') for k,v in encoded[row['example_id']].items()}
        def generate(row):
            inputs=processor.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            inputs={k:v.to('cuda:0') for k,v in inputs.items()}
            with torch.inference_mode():result=model.generate(**inputs,generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
            new=result[0,inputs['input_ids'].shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True)
            eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            return {'example_id':row['example_id'],'split':row['split'],'source':row['source'],'reference':row['target'],
                'prompt_sha256':sha(row['prompt']),'output':text,'output_tokens':int(new.numel()),
                'empty':not bool(text.strip()),'output_limited':bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                'copied_source':' '.join(text.split())==' '.join(row['source'].split())}
        probes=[r for r in groups['training'] if r['example_id'] in config['training_probe_ids']]
        def snapshot(name,step):
            was_training=model.training;model.eval()
            snap={'name':name,'optimizer_step':step,'complete':False,'losses':[],'generations':[]}
            report['snapshots'].append(snap);phase('snapshot_'+name)
            try:
                total=0;count=0
                for row in groups['development']:
                    with torch.inference_mode():loss=float(model(**batch(row)).loss.item())
                    if not math.isfinite(loss):raise RuntimeError('nonfinite_development_loss')
                    n=sum(v!=-100 for v in encoded[row['example_id']]['labels'])
                    snap['losses'].append({'example_id':row['example_id'],'loss':loss,'supervised_tokens':n});total+=loss*n;count+=n
                snap['token_weighted_development_loss']=total/count
                snap['mean_example_development_loss']=sum(r['loss'] for r in snap['losses'])/8
                for row in groups['development']+probes:
                    snap['generations'].append(generate(row));persist()
                    print(name+': saved %d/10 outputs'%len(snap['generations']),flush=True)
                snap['complete']=True;persist()
            finally:model.train(was_training)
        snapshot('base',0)
        if mode=='baseline':
            report['complete']=True;state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-start);persist();return
        report['previous_baseline_sha256']=sha(json.dumps(baseline,sort_keys=True))
        report['previous_baseline_exact_output_matches']=sum(a['output']==b['output'] for a,b in zip(report['snapshots'][0]['generations'],baseline['snapshots'][0]['generations']))
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel,get_peft_model_state_dict
        phase('prepare_adapter');model.config.use_cache=False;model.config.text_config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        matched=[name for name,_ in model.named_modules() if re.fullmatch(TARGET_PATTERN,name)]
        if len(matched)!=2*model.config.text_config.num_hidden_layers:raise RuntimeError('language_module_layout_changed')
        model=get_peft_model(model,LoraConfig(r=HYPER['rank'],lora_alpha=HYPER['alpha'],lora_dropout=HYPER['dropout'],
            target_modules=TARGET_PATTERN,bias='none',task_type='CAUSAL_LM'))
        trainable={k:v for k,v in model.named_parameters() if v.requires_grad}
        if not trainable or any('lora_' not in k or 'language_model' not in k for k in trainable):raise RuntimeError('only_language_adapters_may_train')
        before={k:v.detach().cpu().clone() for k,v in trainable.items()};checks=[]
        def collate(features):
            if len(features)!=1:raise ValueError('batch_size_one_required')
            return {k:torch.tensor([v],dtype=torch.long) for k,v in features[0].items()}
        class Checks(TrainerCallback):
            def on_pre_optimizer_step(self,args,trainer_state,control,**kwargs):
                grads=[p.grad.detach() for p in trainable.values() if p.grad is not None]
                if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('invalid_adapter_gradients')
                norm=math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
                if not math.isfinite(norm) or norm<=0:raise RuntimeError('nonpositive_adapter_gradient')
                checks.append({'step':trainer_state.global_step+1,'norm':norm,'finite':True});write_json(output/'gradient_checks.json',checks)
            def on_epoch_end(self,args,trainer_state,control,**kwargs):
                epoch=round(trainer_state.epoch)
                if not math.isclose(trainer_state.epoch,epoch) or trainer_state.global_step!=epoch*8:raise RuntimeError('epoch_boundary_changed')
                model.save_pretrained(str(output/f'epoch_{epoch}'/'adapter'))
                snapshot('epoch_'+str(epoch),trainer_state.global_step);state['optimizer_steps']=trainer_state.global_step;persist()
                return control
        class FiniteTrainer(Trainer):
            def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
                result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
                loss=result[0] if return_outputs else result
                if not bool(torch.isfinite(loss).all()):raise RuntimeError('nonfinite_training_loss')
                return result
        args=TrainingArguments(output_dir=str(output/'trainer'),num_train_epochs=HYPER['epochs'],
            per_device_train_batch_size=1,gradient_accumulation_steps=2,gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},learning_rate=HYPER['learning_rate'],
            lr_scheduler_type='linear',warmup_steps=0,optim='adamw_torch',fp16=False,bf16=False,
            logging_steps=1,save_strategy='no',report_to='none',seed=HYPER['seed'],data_seed=HYPER['seed'],
            remove_unused_columns=False,prediction_loss_only=True,label_names=['labels'])
        trainer=FiniteTrainer(model=model,args=args,processing_class=processor,data_collator=collate,
            train_dataset=[encoded[r['example_id']] for r in groups['training']],callbacks=[Checks()])
        trainer.model_accepts_loss_kwargs=False
        state['training_called']=True;report['training_called']=True;phase('training');trainer.train()
        changed=any(not torch.equal(before[k],v.detach().cpu()) for k,v in trainable.items())
        if trainer.state.global_step!=16 or len(checks)!=16 or not changed:raise RuntimeError('pilot_updates_or_adapter_change_failed')
        write_json(output/'training_log.json',trainer.state.log_history)
        saved={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        phase('adapter_reload');base=model.unload();model=PeftModel.from_pretrained(base,str(output/'epoch_2/adapter'),is_trainable=False)
        reloaded=get_peft_model_state_dict(model)
        if set(saved)!=set(reloaded) or any(not torch.equal(v,reloaded[k].detach().cpu()) for k,v in saved.items()):raise RuntimeError('adapter_reload_weights_differ')
        model.eval();report['reload_probe']=generate(groups['development'][0]);report['adapter_reload_weights_match']=True
        report['adapter_weights_changed']=True;report['complete']=True
        state.update(complete=True,phase='complete',optimizer_steps=16,adapter_weights_changed=True,
            adapter_reload_weights_match=True,elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Experiment failed; inspect state.json. Error type: '+type(error).__name__) from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=['baseline','pilot'],default='baseline');parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/teacher_pair_base_v1'))
    parser.add_argument('--check',action='store_true');parser.add_argument('--train-authorized',action='store_true')
    args=parser.parse_args(argv);groups,config=load_inputs(args.data,args.config)
    if args.check:print('Validated 16 training / 8 paper- and author-held-out development pairs; no model or training.');return
    baseline=None
    if args.mode=='pilot':
        if not args.train_authorized or args.baseline is None:raise ValueError('explicit_training_authorization_and_baseline_required')
        baseline=validate_baseline(args.baseline,config)
    run_training(groups,config,args.mode,args.output,baseline)


if __name__=='__main__':main()
