"""Frozen-adapter, training-side free generation: no optimization or target prompts."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

PROTOCOL='preference_training_inference_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
ARMS=('sft','dpo')
FIELDS={'pair_id','anchor_group_id','example_id','paper_id','input_kind','family','source','prompt',
        'source_sha256','prompt_sha256','prefix_ids','prefix_ids_sha256'}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(text):return INSTRUCTION+'\n\nPassage:\n'+text
def messages(job):return [{'role':'user','content':[{'type':'text','text':job['prompt']}]}]


def select_rows(rows):
    """Prefer an included original input, otherwise its included variant; no output ranking."""
    groups={}
    for row in rows:groups.setdefault(row['anchor_group_id'],[]).append(row)
    return [sorted(group,key=lambda r:(r['input_kind']!='original',r['pair_id']))[0]
            for _,group in sorted(groups.items())]


def load_inputs(jobs_path,config_path):
    config=json.loads(config_path.read_text())
    if (config.get('protocol')!=PROTOCOL or config.get('model_id')!=MODEL_ID or config.get('model_revision')!=REVISION
            or config.get('decoding')!=DECODING or config.get('arms')!=list(ARMS) or config.get('seed')!=1729
            or config.get('max_length')!=1024 or config.get('maximum_runtime_seconds')!=1800
            or config.get('training_authorized') is not False or config.get('optimizer_steps')!=0
            or config.get('jobs_sha256')!=file_sha(jobs_path) or config.get('runtime_source_sha256')!=file_sha(Path(__file__))):
        raise ValueError('frozen_inference_configuration_changed')
    jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    if (len(jobs)!=11 or len({r['pair_id'] for r in jobs})!=11 or len({r['anchor_group_id'] for r in jobs})!=11
            or len({r['example_id'] for r in jobs})!=11 or len({r['paper_id'] for r in jobs})!=10
            or [r['pair_id'] for r in jobs]!=config['pair_ids']
            or Counter(r['input_kind'] for r in jobs)!=Counter({'original':9,'variant':2})):
        raise ValueError('frozen_eleven_group_selection_changed')
    for row in jobs:
        ids=row['prefix_ids']
        if (set(row)!=FIELDS or not row['source'].strip() or row['prompt']!=prompt(row['source'])
                or sha(row['source'])!=row['source_sha256'] or sha(row['prompt'])!=row['prompt_sha256']
                or not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids)
                or ids[0]!=2 or ids.count(2)!=1 or len(ids)+512>1024
                or sha(json.dumps(ids))!=row['prefix_ids_sha256']):raise ValueError('source_only_prompt_or_prefix_changed')
    if set(config['adapter_sha256'])!=set(ARMS) or any(set(v)!={'adapter_config.json','adapter_model.safetensors'}
            for v in config['adapter_sha256'].values()):raise ValueError('two_exact_epoch_two_adapters_required')
    return jobs,config


def verify_adapter(folder,expected):
    if any(file_sha(folder/name)!=digest for name,digest in expected.items()):raise ValueError('wrong_saved_pilot_adapter')
    c=json.loads((folder/'adapter_config.json').read_text())
    if (c['base_model_name_or_path']!=MODEL_ID or c['r']!=8 or c['lora_alpha']!=16 or c['lora_dropout']!=0.05):
        raise ValueError('saved_adapter_configuration_changed')


def run_training(jobs,config,adapters,output,max_seconds=1800):
    """Repository lazy-ML entrypoint convention; performs inference only."""
    if output.exists():raise ValueError('fresh_inference_output_required')
    for arm in ARMS:verify_adapter(adapters[arm],config['adapter_sha256'][arm])
    output.mkdir(parents=True);started=time.monotonic()
    state={'complete':False,'phase':'start','training_called':False,'optimizer_steps':0,'completed_generations':0}
    report={'protocol':PROTOCOL,'mode':'inference_only','complete':False,'config':config,
        'training_called':False,'optimizer_steps':0,'snapshots':[]}
    def persist():write_json(output/'state.json',state);write_json(output/'evaluation.json',report)
    def budget():
        if time.monotonic()-started>=min(max_seconds,1800):raise TimeoutError('inference_time_budget_exceeded')
    def phase(name):state['phase']=name;persist();print('Phase: '+name,flush=True);budget()
    persist()
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,GenerationConfig,LogitsProcessor,LogitsProcessorList
        from peft import PeftModel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose only GPU 0 on Kaggle T4 x2')
        packages={p:importlib.metadata.version(p) for p in config['packages']}
        if packages!=config['packages']:raise RuntimeError('recorded_pilot_environment_required')
        torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats(0)
        report['runtime']={'packages':packages,'model_id':MODEL_ID,'model_revision':REVISION,
            'gpu':torch.cuda.get_device_name(0),'gpu_count':1,'runtime_source_sha256':file_sha(Path(__file__))}
        phase('model_access')
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
        if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:raise RuntimeError('model_revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        phase('prompt_equivalence');audits=[]
        for job in jobs:
            values=processor.apply_chat_template(messages(job),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            ids=values['input_ids'][0].tolist()
            if ids!=job['prefix_ids'] or ('token_type_ids' in values and bool(values['token_type_ids'].any())):
                raise ValueError('processor_prefix_differs_from_verified_training_ids')
            audits.append({'pair_id':job['pair_id'],'prompt_tokens':len(ids),'prefix_ids_sha256':sha(json.dumps(ids)),'matches_training_prefix':True})
        write_json(output/'token_audit.json',audits)
        phase('load_model')
        base=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;base.eval();base.requires_grad_(False)
        generation=GenerationConfig(**DECODING,eos_token_id=base.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        if generation.to_dict()!=config['expected_generation_config']:raise ValueError('recorded_decoding_configuration_changed')
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('nonfinite_generation_scores')
                return scores
        for arm in ARMS:
            phase('load_'+arm);torch.manual_seed(1729)
            model=PeftModel.from_pretrained(base,str(adapters[arm]),is_trainable=False)
            model.eval();model.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('all_parameters_must_be_frozen')
            snapshot={'arm':arm,'checkpoint_epoch':2,'complete':False,'all_parameters_frozen':True,'generations':[]}
            report['snapshots'].append(snapshot);phase('generate_'+arm)
            for job in jobs:
                budget();state['current_pair_id']=job['pair_id'];persist()
                ids=torch.tensor([job['prefix_ids']],dtype=torch.long,device='cuda:0')
                with torch.inference_mode():result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                    token_type_ids=torch.zeros_like(ids),generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
                new=result[0,ids.shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True)
                eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                snapshot['generations'].append({k:job[k] for k in ('pair_id','example_id','anchor_group_id','paper_id','input_kind','source_sha256','prompt_sha256')}|
                    {'output':text,'output_sha256':sha(text),'output_tokens':int(new.numel()),'empty':not bool(text.strip()),
                     'output_limited':bool(new.numel()>=512 and int(new[-1]) not in eos),
                     'copied_source':' '.join(text.split())==' '.join(job['source'].split())})
                state['completed_generations']+=1;persist();print(arm+': saved %d/11 generations'%len(snapshot['generations']),flush=True)
                del ids,result,new;budget()
            snapshot['complete']=True;persist()
            base=model.unload();del model;base.eval();base.requires_grad_(False)
        if state['completed_generations']!=22:raise ValueError('fixed_generation_coverage_failed')
        for arm in ARMS:verify_adapter(adapters[arm],config['adapter_sha256'][arm])
        report['complete']=True
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Inference stopped; preserve state and partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sft-adapter',type=Path);p.add_argument('--dpo-adapter',type=Path);p.add_argument('--check',action='store_true')
    p.add_argument('--run-inference',action='store_true');p.add_argument('--max-runtime-seconds',type=int,default=1800)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/preference_training_inference_v1'));a=p.parse_args(argv)
    jobs,config=load_inputs(a.jobs,a.config);adapters={'sft':a.sft_adapter,'dpo':a.dpo_adapter}
    for arm,path in adapters.items():
        if path is not None:verify_adapter(path,config['adapter_sha256'][arm])
    if a.check:print('Validated eleven training prompts, two frozen adapters, 22 new generations. No model or training.');return
    if not a.run_inference or any(p is None for p in adapters.values()) or not 0<a.max_runtime_seconds<=1800:
        raise ValueError('explicit_inference_flag_two_adapters_and_bounded_runtime_required')
    run_training(jobs,config,adapters,a.output,a.max_runtime_seconds)


if __name__=='__main__':main()
