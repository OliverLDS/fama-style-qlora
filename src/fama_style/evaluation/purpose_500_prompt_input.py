"""Inference-only 3 input routes × 2 prompts × base/saved adapter diagnostic."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

PROTOCOL='purpose_500_prompt_input_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
PROMPTS={
 'simple':'Rewrite the following passage in natural, human-written academic economics prose.\n\nReturn only the rewritten paragraph.',
 'reexpress':'Rewrite the following passage in natural, human-written academic economics prose. Re-express it using different wording and sentence structure instead of copying or lightly editing the original sentences.\n\nReturn only the rewritten paragraph.'}
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
PACKAGES={'transformers':'5.20.1','peft':'0.21.2','accelerate':'1.15.0','bitsandbytes':'0.50.2','huggingface_hub':'1.33.0'}
MAX_SECONDS=2700
GROUPS=('Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler')
ROUTES=('existing_input','gemini_input','fresh_reconstruction')

def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(source,condition='simple'):return PROMPTS[condition]+'\n\nPassage:\n'+source

def verify_adapter(adapter,report):
    for name,digest in report['adapter_files_sha256'].items():
        if file_sha(adapter/name)!=digest:raise ValueError('adapter_changed')


def load_inputs(report_path,adapter,data,config_path):
    c=json.loads(config_path.read_text())
    fixed=dict(protocol=PROTOCOL,model_id=MODEL_ID,model_revision=REVISION,prompts=PROMPTS,
               decoding=DECODING,packages=PACKAGES,max_runtime_seconds=MAX_SECONDS,
               training_authorized=False,runtime_source_sha256=file_sha(Path(__file__)),dataset_sha256=file_sha(data))
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('frozen_configuration_changed')
    if file_sha(report_path)!=c['parent_report_sha256']:raise ValueError('wrong_parent_notebook_version')
    report=json.loads(report_path.read_text());parent=report['config']
    if report['protocol']!='gpt6_purpose_500_v1' or not report['complete'] or report['optimizer_steps']!=250 or report['training_examples_seen']!=500 or not report['adapter_reload_weights_match']:
        raise ValueError('completed_parent_required')
    if parent['model_id']!=MODEL_ID or parent['model_revision']!=REVISION or parent['packages']!=PACKAGES or parent['decoding']!=DECODING:raise ValueError('parent_recipe_changed')
    if [s['name'] for s in report['snapshots']]!=['base','epoch_1'] or any(not s['complete'] for s in report['snapshots']):raise ValueError('parent_snapshots_incomplete')
    verify_adapter(adapter,report)
    original={r['example_id']:r for r in report['snapshots'][0]['generations']}
    selected=c['selected_content_ids']
    if len(selected)!=4 or len(set(selected))!=4 or {original[x]['style_group'] for x in selected}!=set(GROUPS):raise ValueError('four_author_balanced_contents_required')
    rows=[json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    if len(rows)!=24 or [r['example_id'] for r in rows]!=c['job_ids'] or len({r['example_id'] for r in rows})!=24:raise ValueError('job_coverage')
    seen=set()
    for r in rows:
        cid=r['content_id'];route=r['input_condition'];condition=r['prompt_condition']
        if cid not in selected or route not in ROUTES or condition not in PROMPTS:raise ValueError('unknown_condition')
        identity=(cid,route,condition)
        if identity in seen:raise ValueError('duplicate_condition')
        seen.add(identity);old=original[cid]
        if r['example_id']!=cid+'::'+route+'::'+condition:raise ValueError('job_identity_changed')
        if any(r[k]!=old[k] for k in ['paper_id','style_group','reference']):raise ValueError('reference_or_paper_changed')
        if r['source_sha256']!=sha(r['source']) or r['reference_sha256']!=sha(r['reference']) or r['prompt_sha256']!=sha(prompt(r['source'],condition)):raise ValueError('text_or_prompt_hash')
        if not isinstance(r['source'],str) or not r['source'].strip():raise ValueError('empty_source')
        if route!='fresh_reconstruction':
            previous=original[cid if route=='existing_input' else cid+'::gemini_input_v1']
            if r['source']!=previous['source']:raise ValueError('existing_control_changed')
        elif r['source_sha256']!=c['fresh_source_sha256_by_content'][cid]:raise ValueError('fresh_input_changed')
        if r['canonical_pdf_sha256']!=parent['canonical_pdf_by_paper'][r['paper_id']]:raise ValueError('PDF_binding_changed')
    if len(seen)!=24:raise ValueError('factorial_coverage')
    # The same route text must be shared across the two prompts.
    for cid in selected:
        for route in ROUTES:
            variants=[r for r in rows if r['content_id']==cid and r['input_condition']==route]
            if len({r['source_sha256'] for r in variants})!=1:raise ValueError('prompt_pair_source_changed')
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
            'instructions':PROMPTS,'decoding':DECODING,'snapshots':[],
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
            text=prompt(row['source'],row['prompt_condition']);messages=[{'role':'user','content':[{'type':'text','text':text}]}]
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
                    prompt_sha256=sha(prompt(row['source'],row['prompt_condition'])),output_tokens=int(new.numel()),empty=not bool(text.strip()),
                    output_limited=bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                    copied_source=' '.join(text.split())==' '.join(row['source'].split())))
                state['generations']+=1;persist();print(name+': saved %d/24 outputs'%len(snap['generations']),flush=True)
            snap['complete']=True;persist()
        snapshot('base');phase('load_frozen_adapter');verify_adapter(adapter,original)
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
        snapshot('epoch_1');verify_adapter(adapter,original)
        if state['generations']!=48:raise RuntimeError('generation_coverage_failed')
        result.update(complete=True,adapter_hashes_unchanged=True,all_weights_frozen=True)
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0))
        persist();print('Complete: 48 input-route/prompt outputs saved; zero training updates.',flush=True)
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Inference stopped; preserve state.json and partial results. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['report','adapter','data','config']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--max-runtime-seconds',type=int,default=MAX_SECONDS)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/purpose_500_prompt_input_v1'))
    a=p.parse_args(argv);rows,report=load_inputs(a.report,a.adapter,a.data,a.config)
    if a.check:
        print('Validated 4 held-out contents × 3 input routes × 2 prompts; saved adapter only. No model loaded.');return
    if not 0<a.max_runtime_seconds<=MAX_SECONDS:raise ValueError('bounded_inference_time_required')
    run_training(rows,report,a.adapter,a.output,a.max_runtime_seconds)


if __name__=='__main__':main()
