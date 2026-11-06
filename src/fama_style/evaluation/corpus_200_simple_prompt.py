"""Inference-only prompt comparison using the frozen 200-pair adapter."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

PROTOCOL='corpus_200_simple_prompt_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
INSTRUCTION='''Rewrite the following passage in natural, human-written academic economics prose.

Return only the rewritten paragraph.'''
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
PACKAGES={'transformers':'5.20.1','peft':'0.21.2','accelerate':'1.15.0',
          'bitsandbytes':'0.50.2','huggingface_hub':'1.33.0'}
MAX_SECONDS=1800


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(source):return INSTRUCTION+'\n\nPassage:\n'+source


def verify_adapter(adapter,report):
    for name in ['adapter_config.json','adapter_model.safetensors']:
        if file_sha(adapter/name)!=report['adapter_files_sha256'][name]:raise ValueError('adapter_changed')


def load_inputs(report_path,adapter,expected_sha256):
    if file_sha(report_path)!=expected_sha256:raise ValueError('attach_exact_pilot_version_1_output')
    report=json.loads(report_path.read_text());config=report['config']
    if (report.get('protocol')!='corpus_200_pilot_v1' or report.get('complete') is not True
            or report.get('training_examples_seen')!=200 or report.get('optimizer_steps')!=100
            or report.get('adapter_reload_weights_match') is not True
            or config.get('model_id')!=MODEL_ID or config.get('model_revision')!=REVISION
            or config.get('decoding')!=DECODING or config.get('packages')!=PACKAGES):
        raise ValueError('completed_frozen_200_pair_pilot_required')
    if [s['name'] for s in report['snapshots']]!=['base','epoch_1']:raise ValueError('both_original_conditions_required')
    old={}
    for snap in report['snapshots']:
        if not snap['complete'] or [r['example_id'] for r in snap['generations']]!=config['development_ids']:
            raise ValueError('original_generation_coverage')
        old[snap['name']]={r['example_id']:r for r in snap['generations']}
    rows=[]
    for rid in config['development_ids']:
        original=old['base'][rid];other=old['epoch_1'][rid]
        if any(original[k]!=other[k] for k in ['source','reference','paper_id','style_group']):
            raise ValueError('original_matched_inputs_differ')
        rows.append({k:original[k] for k in ['example_id','source','reference','paper_id','style_group']})
    if len(rows)!=16 or len({r['example_id'] for r in rows})!=16 or len({r['paper_id'] for r in rows})!=8:
        raise ValueError('same_sixteen_held_out_examples_required')
    if sorted(Counter(r['style_group'] for r in rows).values())!=[4,4,4,4]:raise ValueError('group_balance_changed')
    detector=config['detector_case_ids']
    if len(detector)!=8 or len(set(detector))!=8 or not set(detector)<=set(config['development_ids']):
        raise ValueError('original_detector_cases_required')
    verify_adapter(adapter,report)
    return rows,report


def run_training(rows,original,adapter,output,max_seconds):
    """Historical GPU-entry-point name; this function performs inference only.

    All ML imports stay inside this entry point. No optimizer, backward pass,
    new adapter initialization or checkpoint write is implemented.
    """
    if output.exists():raise ValueError('fresh_output_directory_required')
    output.mkdir(parents=True);started=time.monotonic()
    state={'protocol':PROTOCOL,'phase':'start','complete':False,'training_called':False,'optimizer_steps':0,'generations':0}
    result={'protocol':PROTOCOL,'complete':False,'training_called':False,'optimizer_steps':0,
            'instruction':INSTRUCTION,'decoding':DECODING,'snapshots':[],
            'previous_detector_case_ids':original['config']['detector_case_ids'],
            'previous_development_ids':original['config']['development_ids'],
            'previous_adapter_files_sha256':original['adapter_files_sha256'],
            'semantic_assessment':None,'detector_results':None}
    def persist():
        write_json(output/'state.json',state);write_json(output/'evaluation.json',result)
    def budget():
        if time.monotonic()-started>=min(max_seconds,MAX_SECONDS):raise TimeoutError('inference_time_cap')
    def phase(name):
        state['phase']=name;persist();budget();print('Phase: '+name,flush=True)
    persist()
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,GenerationConfig,LogitsProcessor,LogitsProcessorList
        from peft import PeftModel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose GPU 0 only on Kaggle T4 x2')
        packages={name:importlib.metadata.version(name) for name in PACKAGES}
        if packages!=PACKAGES or str(torch.__version__)!=original['runtime']['packages']['torch']:
            raise RuntimeError('same_recorded_runtime_required_for_prompt_comparison')
        torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats(0)
        result['runtime']={'model_id':MODEL_ID,'model_revision':REVISION,'packages':dict(packages,torch=str(torch.__version__)),
                           'gpu':torch.cuda.get_device_name(0),'gpu_count':1,'runtime_source_sha256':file_sha(Path(__file__))}
        phase('model_access')
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
        if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:
            raise RuntimeError('model_revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        phase('prompt_preflight');encoded={};audits=[]
        for row in rows:
            # References are retained solely for later evaluation, never messages.
            text=prompt(row['source']);messages=[{'role':'user','content':[{'type':'text','text':text}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            n=inputs['input_ids'].shape[-1]
            if n+DECODING['max_new_tokens']>1536:raise ValueError('no_truncation_context_budget')
            encoded[row['example_id']]=inputs
            audits.append({'example_id':row['example_id'],'source_sha256':sha(row['source']),
                           'prompt_sha256':sha(text),'prompt_tokens':n,'reference_in_messages':False,'truncated':False})
        write_json(output/'prompt_audit.json',audits)
        phase('load_base')
        model=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None
        generation=GenerationConfig(**DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('nonfinite_generation_scores')
                return scores
        def snapshot(name):
            model.eval();model.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('inference_weights_must_be_frozen')
            snap={'name':name,'complete':False,'generations':[]};result['snapshots'].append(snap);phase('generate_'+name)
            for row in rows:
                budget();state['current_example_id']=row['example_id'];persist()
                inputs={k:v.to('cuda:0') for k,v in encoded[row['example_id']].items()}
                with torch.inference_mode():generated=model.generate(**inputs,generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
                new=generated[0,inputs['input_ids'].shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True)
                eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                snap['generations'].append(dict(row,output=text,output_sha256=sha(text),source_sha256=sha(row['source']),
                    prompt_sha256=sha(prompt(row['source'])),output_tokens=int(new.numel()),empty=not bool(text.strip()),
                    output_limited=bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                    copied_source=' '.join(text.split())==' '.join(row['source'].split())))
                state['generations']+=1;persist();print(name+': saved %d/16 outputs'%len(snap['generations']),flush=True)
            snap['complete']=True;persist()
        snapshot('base');phase('load_frozen_adapter');verify_adapter(adapter,original)
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
        snapshot('epoch_1');verify_adapter(adapter,original)
        if state['generations']!=32:raise RuntimeError('generation_coverage_failed')
        result.update(complete=True,adapter_hashes_unchanged=True,all_weights_frozen=True)
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0))
        persist();print('Complete: 32 simpler-prompt outputs saved; zero training updates.',flush=True)
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Inference stopped; preserve state.json and partial results. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--report-sha256',required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--max-runtime-seconds',type=int,default=MAX_SECONDS)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/corpus_200_simple_prompt_v1'))
    a=p.parse_args(argv);rows,report=load_inputs(a.report,a.adapter,a.report_sha256)
    if a.check:
        print('Validated 16 unchanged held-out inputs and exact saved adapter. No model loaded or training.');return
    if not 0<a.max_runtime_seconds<=MAX_SECONDS:raise ValueError('bounded_inference_time_required')
    run_training(rows,report,a.adapter,a.output,a.max_runtime_seconds)


if __name__=='__main__':main()
