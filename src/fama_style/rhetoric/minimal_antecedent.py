"""Source-supported antecedent expansion versus frozen context controls."""
import argparse
from collections import Counter
import copy
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import time
from . import groupwise_realization_v2 as parent
from . import group_context_restoration as previous
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha
from fama_style.research.signal import features

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='minimal_antecedent_v1'
DECODING=parent.DECODING
SEEDS=parent.SEEDS
SELECTION=(('F1980::FAMA1980_0028',2),('FF2000::FF00-DD-0070',4),('FF2000::FF00-DD-0070',5))
CONDITIONS=('A_local_only','B_read_only_context','C_explicit_actors')
PARENT_SHA='ed0699f3b413ff94911df44925214b93d332b5ce3346dcbf3155f9f610ab03b5'
EDIT_SPEC={
    ('F1980::FAMA1980_0028',2):[("Their", "Managers'", ['Managers'])],
    ('FF2000::FF00-DD-0070',4):[
        ('6.8 percent expected to start','6.8 percent of firms that have never paid dividends expected to start paying dividends',['firms that have never paid dividends','paying']),
        ('3.0 percent actually starting','3.0 percent actually starting to pay dividends',['dividends'])],
    ('FF2000::FF00-DD-0070',5):[
        ('5.2 percent were predicted to start','5.2 percent of firms that have never paid dividends were predicted to start paying dividends',['firms that have never paid dividends','paying']),
        ('0.8 percent did','0.8 percent of those firms did',['firms that have never paid dividends'])]
}


def source_group(row,index):
    return ' '.join(u['text'] for u in row['source_units'] if u['unit_id'] in row['groups'][index-1])


def expansion(row,index):
    original=source_group(row,index);text=original;edits=[]
    for before,after,support in EDIT_SPEC[(row['id'],index)]:
        if text.count(before)!=1:raise ValueError('Ambiguous expansion anchor')
        citations=[]
        for quote in support:
            start=row['source'].find(quote)
            if start<0:raise ValueError('Missing source support')
            citations.append({'quote':quote,'start':start,'end':start+len(quote)})
        edits.append({'before':before,'after':after,'source_support':citations});text=text.replace(before,after,1)
    if re.findall(r'\d+(?:\.\d+)?',text)!=re.findall(r'\d+(?:\.\d+)?',original):raise ValueError('Changed numeric values')
    return {'original_group_sha256':sha(original),'expanded_group':text,'expanded_group_sha256':sha(text),
        'edits':edits,'provenance':'Deterministic, experiment-specific in-session antecedent expansion from plain-English source only. Not a learned coreference resolver or independently verified annotation.'}


def jobs_for(pack):
    controls=previous.jobs_for(pack);rows={r['id']:r for r in pack['rows']};result=[]
    for i in range(0,len(controls),2):
        pair=controls[i:i+2];a=next(j for j in pair if j['condition']==CONDITIONS[0]);r=rows[a['id']]
        prefix,assigned=a['prompt'].split('Assigned group:\n',1)
        if assigned!=source_group(r,a['group_index']):raise ValueError('Changed assignment')
        prompt=prefix+'Assigned group:\n'+expansion(r,a['group_index'])['expanded_group']
        c={**a,'condition':CONDITIONS[2],'prompt':prompt,'prompt_sha256':sha(prompt)}
        triple=sorted(pair+[c],key=lambda j:j['condition'])
        offset=(i//2)%3;result.extend(triple[offset:]+triple[:offset])
    return result


def job_key(job):return parent.job_key(job)
def runtime_paths():return sorted(set(previous.runtime_paths()+[Path(previous.__file__).resolve(),Path(__file__).resolve()]))
def audit_tokens(job,ids):return parent.audit_tokens(job,ids)


def prepare_pack(output):
    root=ROOT/'research_runs/group_context_restoration_v1'
    old=previous.load_pack(root/'input_pack/context_pack.json',PARENT_SHA)
    reviewed=json.loads((root/'version_1_review/verification.json').read_text())
    if not all(reviewed['checks'].values()):raise ValueError('Parent review failed')
    run=root/'kaggle_version_1/group_context_restoration'
    for name,h in reviewed['artifact_hashes'].items():
        if file_sha(run/name)!=h:raise ValueError('Parent artifacts changed')
    pack=copy.deepcopy(old);pack.update(protocol=PROTOCOL,parent_pack_sha256=PARENT_SHA)
    pack['historical_context_controls']=[{**{k:r[k] for k in ('id','condition','seed','group_index','prompt_sha256','sampling_seed','max_new_tokens')},'output_sha256':sha(r['output'])} for r in read_lines(run/'generations.private.jsonl')]
    pack['expansions']=[{'id':r['id'],'group_index':i,**expansion(r,i)} for r in pack['rows'] for i in r['selected_group_indices']]
    pack['scope']='Three source groups, three matched seeds, local-only and full-context controls versus explicit actor/cohort expansion. No reference wording in prompts, assembly, learned resolver or training.'
    pack['runtime_code_files']={str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False));validate_controls(pack)
    fresh(output);write_json(output/'antecedent_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','generation_calls':len(jobs_for(pack)),'pack_sha256':file_sha(output/'antecedent_pack.json'),'training_called':False})
    print('Prepared 27 calls: 18 exact controls and 9 explicit-actor candidates. No inference executed.')


def validate_controls(pack):
    expected={job_key(r):r for r in pack['historical_context_controls']}
    controls=[j for j in jobs_for(pack) if j['condition'] in CONDITIONS[:2]]
    if len(controls)!=18 or set(expected)!={job_key(j) for j in controls}:raise ValueError('Wrong control identities')
    for j in controls:
        if any(j[k]!=expected[job_key(j)][k] for k in ('prompt_sha256','sampling_seed','max_new_tokens')):raise ValueError('Changed historical control')


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if (pack['protocol']!=PROTOCOL or pack['parent_pack_sha256']!=PARENT_SHA or pack['model_id']!=MODEL_ID
        or pack['model_revision']!=MODEL_REVISION or pack['train_sha256']!=TRAIN_SHA or pack['decoding']!=DECODING):raise ValueError('Changed experiment identity')
    selected={(r['id'],i) for r in pack['rows'] for i in r['selected_group_indices']}
    if selected!=set(SELECTION):raise ValueError('Changed selection')
    for r in pack['rows']:parent.prior.validate_grouping(r['source'],r['reference'],r['groups'],r['source_sha256'],r['reference_sha256'])
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Changed jobs')
    if pack['expansions']!=[{'id':r['id'],'group_index':i,**expansion(r,i)} for r in pack['rows'] for i in r['selected_group_indices']]:raise ValueError('Changed expansion provenance')
    validate_controls(pack);return pack


def evaluate(pack,outputs,evidence=None,synthetic=False):
    jobs={job_key(j):j for j in jobs_for(pack)};rows={r['id']:r for r in pack['rows']};seen=set();records=[]
    for out in outputs:
        key=job_key(out)
        if key in seen or key not in jobs:raise ValueError('Duplicate or unknown call')
        seen.add(key);j=jobs[key];row=rows[j['id']]
        if any(out[k]!=j[k] for k in ('prompt_sha256','sampling_seed','max_new_tokens')):raise ValueError('Changed generation identity')
        if out.get('synthetic') is not synthetic or type(out['hit_output_limit']) is not bool:raise ValueError('Invalid generation flags')
        available=bool(out['output'].strip()) and not out['hit_output_limit']
        group=' '.join(u['text'] for u in row['source_units'] if u['unit_id'] in row['groups'][j['group_index']-1])
        verdict=(evidence or {}).get(key)
        if available and not synthetic:
            if not verdict or verdict['output_sha256']!=sha(out['output']) or verdict['source_group_sha256']!=sha(group) or verdict['context_sha256']!=sha(row['source']) or verdict.get('expanded_group_sha256')!=expansion(row,j['group_index'])['expanded_group_sha256']:raise ValueError('Missing or unbound diagnostics')
        historical=next((r for r in pack.get('historical_context_controls',[]) if job_key(r)==key),None)
        records.append({**out,'available':available,'format':parent.piece_format(out['output']),
            'output_sha256':sha(out['output']),'diagnostics':verdict if available else None,
            'output_features':features(out['output']) if available else None,
            'source_similarity':__import__('difflib').SequenceMatcher(None,group,out['output']).ratio() if available else None,
            'historical_control_output_matches':sha(out['output'])==historical['output_sha256'] if historical and j['condition'] in CONDITIONS[:2] else None})
    if seen!=set(jobs):raise ValueError('Incomplete calls')
    conditions={}
    for condition in CONDITIONS:
        items=[r for r in records if r['condition']==condition]
        conditions[condition]={'attempts':len(items),'available':sum(r['available'] for r in items),
            'single_sentence_complete':sum(r['available'] and r['format']['single_sentence'] and r['format']['terminal_punctuation'] for r in items),
            'historical_control_output_matches':sum(r['historical_control_output_matches'] is True for r in items) if condition in CONDITIONS[:2] else None,
            **{k:sum(r.get(k,0) for r in items) for k in ('input_tokens','output_tokens','generation_seconds')}}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete' if all(r['available'] for r in records) else 'execution_complete_with_generation_failures',
        'conditions':conditions,'training_called':False,'reserved_test_rows_read':0,'validation_rows_read':0,
        'semantic_accuracy':None,'author_style_accuracy':None,'qualitative_assessment':'pending',
        'limitations':'Three selected training groups, not generalization evidence. Explicit actors/cohorts are deterministic experiment-specific expansions, not learned resolution. A/C differs only in assigned text; B also contains context-use instructions. Local and expanded-group NLI plus context entailment are uncalibrated diagnostics, not severity grades. No paragraph assembly or validated author-style metric.'}

def run_training(pack,output,pack_sha):
    """Explicit inference entry point; all ML imports stay within this function."""
    fresh(output);jobs=jobs_for(pack);outputs=[];started=time.monotonic()
    status={'protocol':PROTOCOL,'complete':False,'training_called':False,'completed_jobs':0,'total_jobs':len(jobs)}
    def phase(name):
        status.update(phase=name,completed_jobs=len(outputs));write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        phase('loading_gemma_runtime')
        import torch
        from transformers import AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one T4')
        torch.cuda.init();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select T4 x2')
        token=secret_token()
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        def tokenize(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],tokenize=True,
                add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[audit_tokens(j,tokenize(j)['input_ids'][0].tolist()) for j in jobs];write_json(output/'token_audit.json',audits)
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token,
            dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):raise RuntimeError('Unexpected placement')
        model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('Nonfinite scores')
                return scores
        for job,audit in zip(jobs,audits):
            phase('generating_'+str(len(outputs)+1)+'_of_'+str(len(jobs)));torch.manual_seed(job['sampling_seed'])
            batch=tokenize(job);ids=batch['input_ids'][0].tolist()
            if audit_tokens(job,ids)!=audit:raise ValueError('Tokenizer changed')
            batch={k:v.to('cuda:0') for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,**{k:v for k,v in DECODING.items() if k not in ('context_limit','max_new_tokens')},
                    max_new_tokens=job['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            out={**{k:job[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},'output':text,'synthetic':False,
                'hit_output_limit':bool(new.numel()>=job['max_new_tokens'] and int(new[-1]) not in eos),
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before}
            out['generation_failure']='token_cutoff' if out['hit_output_limit'] else 'empty_output' if not text else None
            outputs.append(out);write_lines(output/'generations.private.jsonl',outputs)
            status.update(completed_jobs=len(outputs),failed_generation_calls=sum(r['generation_failure'] is not None for r in outputs))
            if out['generation_failure']:print('Recorded failed sample: '+out['generation_failure'],flush=True)
        generation_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize;gc.collect();torch.cuda.empty_cache()
        phase('loading_semantic_diagnostics')
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False,dtype=torch.float32).to('cuda:0'),'NLI model')
        nli.eval();nli.requires_grad_(False);labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Wrong NLI labels')
        limit=min(512,int(tokenizer.model_max_length),int(getattr(nli.config,'max_position_embeddings',512)));cache={}
        def predict(a,b):
            key=(sha(a),sha(b))
            if key in cache:return cache[key]
            batch=tokenizer(a,b,truncation=False,return_tensors='pt')
            if batch['input_ids'].shape[-1]>limit:cache[key]=None;return None
            with torch.inference_mode():p=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
            cache[key]={labels[i]:float(x) for i,x in enumerate(p)};return cache[key]
        rows={r['id']:r for r in pack['rows']};evidence={};verdict_rows=[]
        for i,candidate in enumerate(outputs):
            phase('diagnostics_'+str(i+1)+'_of_'+str(len(outputs)))
            if candidate['hit_output_limit'] or not candidate['output'].strip():continue
            row=rows[candidate['id']]
            group=' '.join(u['text'] for u in row['source_units'] if u['unit_id'] in row['groups'][candidate['group_index']-1])
            diagnostic={**{k:candidate[k] for k in ('id','condition','seed','group_index')},
                'output_sha256':sha(candidate['output']),'source_group_sha256':sha(group),'context_sha256':sha(row['source']),
                'local_group_verifier':verifier.verify(group,candidate['output'],predict),
                'full_context_to_output_nli':predict(row['source'],candidate['output']),
                'expanded_group_sha256':expansion(row,candidate['group_index'])['expanded_group_sha256'],
                'expanded_group_verifier':verifier.verify(expansion(row,candidate['group_index'])['expanded_group'],candidate['output'],predict)}
            evidence[job_key(candidate)]=diagnostic;verdict_rows.append(diagnostic)
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
        records,summary=evaluate(pack,outputs,evidence)
        write_lines(output/'jobs.private.jsonl',jobs);write_lines(output/'metrics.private.jsonl',records);write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_model':NLI_MODEL,'nli_revision':NLI_REVISION,
            'decoding':DECODING,'elapsed_seconds':time.monotonic()-started,'generation_memory':generation_memory,
            'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True,all_candidates_complete=all(r['available'] for r in records));phase(summary['status'])
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',completed_jobs=len(outputs),exception_type=type(exc).__name__);write_json(output/'run_status.json',status)
        print('Inference failed; exception details suppressed to protect credentials.',flush=True)
        raise RuntimeError('Inference incomplete; inspect saved status and partial artifacts') from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument('--prepare-pack',action='store_true');a.add_argument('--dry-run',action='store_true');a.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.pack is None:p.error('--pack required (also for synthetic dry run)')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha);return
    fresh(args.output_dir)
    if args.dry_run:
        outputs=[]
        for j in jobs_for(pack):
            outputs.append({**{k:j[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},'output':'Synthetic placeholder sentence.','synthetic':True,'hit_output_limit':False})
        _,summary=evaluate(pack,outputs,synthetic=True);write_json(args.output_dir/'synthetic_evaluation.json',summary)
    else:write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'status':'preflight_passed','calls':len(jobs_for(pack)),'model_calls':0,'training_called':False})
    print('CPU check passed; no Secret, model weights or GPU accessed.')


if __name__=='__main__':main()
