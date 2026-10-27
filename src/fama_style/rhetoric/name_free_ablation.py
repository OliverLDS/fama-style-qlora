"""Author-name instruction versus a concrete name-free style specification; inference only."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import soft_guidance_experiment as parent
from . import style_tradeoff as previous
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,tokens
from fama_style.research.signal import features
from fama_style.research.style_drift import score_candidate,frontier

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='name_free_style_ablation_v1'
SEEDS=(41,42,43)
CONDITIONS=('A_author_name','B_concrete_style')
DECODING={'do_sample':True,'temperature':0.6,'top_p':0.9,'top_k':0,'max_new_tokens':768,'context_limit':4096}
POLICY=Path('docs/style_drift_tradeoff_v1/policy.json')
AUTHOR_NAME_STYLE='Rewrite the passage in the academic economics style of Eugene Fama and his coauthors. '
COMMON_INSTRUCTION=previous.INSTRUCTION.removeprefix(AUTHOR_NAME_STYLE)
CONCRETE_STYLE=(
    'Rewrite the passage as direct, analytical economics prose. '
    'Open with the substantive claim or issue rather than a broad statement about the literature. '
    'Use existing economic actors and variables as subjects and direct verbs for methods and findings. '
    'Make the existing comparisons, conditions, exceptions and inferences easy to follow, without imposing a fixed argument order. '
    'Keep necessary technical terms and place qualifications close to the claims they qualify. '
    'Use economical wording rather than inflated nominal phrases. '
    'End with the last substantive claim rather than a generic statement about complexity or a new research agenda. '
    'No fixed sentence length or sentence pattern is required. ')


def jobs_for(pack):
    rows=pack['rows']
    if len(rows)!=4 or len({r['id'] for r in rows})!=4 or any(r['split']!='validation' or not r['source'].strip() for r in rows):
        raise ValueError('Expected the same four unique development inputs')
    result=[]
    for index,r in enumerate(rows):
        for seed in SEEDS:
            for condition in CONDITIONS if index%2==0 else CONDITIONS[::-1]:
                head=AUTHOR_NAME_STYLE if condition=='A_author_name' else CONCRETE_STYLE
                prompt=head+COMMON_INSTRUCTION+'\n\nPassage:\n'+r['source']
                result.append({'id':r['id'],'condition':condition,'seed':seed,'prompt':prompt,
                    'prompt_sha256':sha(prompt),'source_sha256':sha(r['source']),'oracle':False})
    return result


def runtime_paths():
    return parent.runtime_paths()+[ROOT/POLICY]


def prepare_pack(output):
    root=ROOT/'research_runs/style_tradeoff_candidates_v1'
    path=root/'input_pack/style_tradeoff_pack.json';release=json.loads((root/'release_verification.json').read_text())
    old=previous.load_pack(path,release['pack_sha256'])
    review=json.loads((root/'version_1_review/verification.json').read_text());run=root/'kaggle_version_1/style_tradeoff_candidates_v1'
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Prior run was not verified')
    for name,h in review['artifact_hashes'].items():
        if file_sha(run/name)!=h:raise ValueError('Prior artifacts changed')
    controls=[r for r in read_lines(run/'generations.private.jsonl') if r['condition']=='A_direct']
    if len(controls)!=12:raise ValueError('Missing historical direct controls')
    expected=[{'id':r['id'],'seed':r['seed'],'prompt_sha256':r['prompt_sha256'],'output_sha256':sha(r['output'])} for r in controls]
    training_path=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(training_path)!=TRAIN_SHA:raise ValueError('Training reference provenance changed')
    training=read_lines(training_path)
    if len(training)!=247 or any(r['split']!='train' for r in training):raise ValueError('Wrong reference evidence split')
    inspected=[]
    for paper in sorted({r['paper_key'] for r in training}):
        group=sorted((r for r in training if r['paper_key']==paper),key=lambda r:(len(tokens(r['completion'])),r['id']))
        r=group[(len(group)-1)//2]
        inspected.append({'paper_key':paper,'id':r['id'],'reference_sha256':sha(r['completion'])})
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'train_sha256':TRAIN_SHA,'validation_sha256':parent.VALIDATION_SHA,
        'rows':old['rows'],'style_profile':old['style_profile'],'policy':old['policy'],'decoding':DECODING,
        'historical_author_controls':expected,'parent_pack_sha256':file_sha(path),
        'selection':{'rule':'Reuse all four inputs from verified style-tradeoff v1; no new subset selection.',
                     'previously_exposed_development':True,'reserved_test_rows_read':0},
        'style_specification':{'author_name':AUTHOR_NAME_STYLE,'concrete_name_free':CONCRETE_STYLE,
            'common_semantic_and_output_instruction':COMMON_INSTRUCTION,
            'reference_inspection':inspected,
            'provenance':'Designed after the previous development run; informed by seven median-length training-reference paragraphs (one per training paper), plus the existing train-only profile. Not automatically learned or independently validated.',
            'estimand':'Compare two style-instruction strategies; changing specificity as well as author naming does not isolate the causal effect of the name alone.'},
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}}
    historical={(r['id'],r['seed']):r for r in expected}
    if any(j['prompt_sha256']!=historical[(j['id'],j['seed'])]['prompt_sha256'] for j in jobs_for(pack) if j['condition']=='A_author_name'):
        raise ValueError('Author-name control must exactly reproduce prior prompts')
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))
    fresh(output);write_json(output/'name_free_style_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':4,'jobs':24,
        'pack_sha256':file_sha(output/'name_free_style_pack.json'),'training_called':False,'reserved_test_rows_read':0})
    print('Prepared name-free instruction ablation: four cases / 24 jobs; no models accessed.');return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if (pack['protocol']!=PROTOCOL or pack['model_id']!=MODEL_ID or pack['model_revision']!=MODEL_REVISION
        or pack['train_sha256']!=TRAIN_SHA or pack['validation_sha256']!=parent.VALIDATION_SHA
        or pack['decoding']!=DECODING):raise ValueError('Changed experiment identity')
    if pack['policy']!=json.loads((ROOT/POLICY).read_text()):raise ValueError('Changed scoring policy')
    if pack['style_profile']['fit_split']!='train' or pack['style_profile']['training_rows']!=247:raise ValueError('Non-training style profile')
    spec=pack['style_specification']
    if (spec['author_name']!=AUTHOR_NAME_STYLE or spec['concrete_name_free']!=CONCRETE_STYLE
        or spec['common_semantic_and_output_instruction']!=COMMON_INSTRUCTION):raise ValueError('Changed style specification')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Changed jobs')
    return pack


def evaluate(pack,outputs,semantic=None,synthetic=False):
    jobs=jobs_for(pack);expected={(j['id'],j['condition'],j['seed']):j for j in jobs};seen=set();records=[]
    rows={r['id']:r for r in pack['rows']}
    if len(outputs)!=len(jobs):raise ValueError('Incomplete output set')
    if not synthetic and set(semantic or {})!=set(expected):raise ValueError('Missing semantic evidence')
    for out in outputs:
        key=(out['id'],out['condition'],out['seed']);j=expected.get(key)
        if key in seen or j is None or out['prompt_sha256']!=j['prompt_sha256']:raise ValueError('Duplicate or swapped output')
        seen.add(key)
        if not out['output'].strip() or out.get('synthetic') is not synthetic or out['hit_output_limit'] is not False:raise ValueError('Invalid generation')
        row=rows[out['id']];scores=score_candidate(row['source'],out['output'],pack['style_profile'],(semantic or {}).get(key),pack['policy'])
        records.append({'id':out['id'],'condition':out['condition'],'seed':out['seed'],'candidate_id':sha(out['id']+'|'+out['output'])[:20],
            'output_sha256':sha(out['output']),'format_valid':parent.paragraph_format(out['output']),
            'output_features':features(out['output']),'verifier':(semantic or {}).get(key),
            'historical_control_output_matches':next((r['output_sha256']==sha(out['output']) for r in pack.get('historical_author_controls',[]) if r['id']==out['id'] and r['seed']==out['seed']),None) if out['condition']=='A_author_name' else None,**scores})
    # Do not select winners using references or discard semantic warnings. Show all candidates.
    fronts={r['id']:frontier([{'candidate_id':'source-control','style_alignment_benefit_proxy':0,'semantic_cost_proxy':0}]+
        [m for m in records if m['id']==r['id'] and m['format_valid']]) for r in pack['rows']}
    counts={c:{'outputs':sum(r['condition']==c for r in records),
        'unique_paragraphs':len({r['output_sha256'] for r in records if r['condition']==c}),
        'mean_style_benefit_proxy':statistics.mean(r['style_alignment_benefit_proxy'] for r in records if r['condition']==c),
        'verifier_statuses':dict(Counter(r['old_verifier_status'] for r in records if r['condition']==c)),
        'historical_control_output_matches':sum(r['historical_control_output_matches'] is True for r in records if r['condition']==c) if c=='A_author_name' else None} for c in CONDITIONS}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete',
        'conditions':counts,'within_case_frontiers':fronts,'training_called':False,'reserved_test_rows_used':0,
        'qualitative_style_assessment':'pending','semantic_accuracy':None,'author_style_accuracy':None,
        'limitations':'Four reused development cases, post-hoc instruction design. This compares style-instruction strategies, not the causal effect of author naming alone. Proxies are not calibrated semantic severity or author-style accuracy; all 24 outputs require qualitative assessment.'}


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256'] or not ids or len(ids)+DECODING['max_new_tokens']>DECODING['context_limit']:
        raise ValueError('Changed or over-context prompt; no truncation')
    return {'id':job['id'],'condition':job['condition'],'seed':job['seed'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))}


def run_training(pack,output,pack_sha):
    """Explicit inference entry point; all ML imports stay within this function."""
    fresh(output);jobs=jobs_for(pack);outputs=[];started=time.monotonic()
    status={'protocol':PROTOCOL,'complete':False,'training_called':False,'completed_jobs':0,'total_jobs':24}
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
            phase('generating_'+str(len(outputs)+1)+'_of_24');torch.manual_seed(job['seed'])
            batch=tokenize(job);ids=batch['input_ids'][0].tolist()
            if audit_tokens(job,ids)!=audit:raise ValueError('Tokenizer changed')
            batch={k:v.to('cuda:0') for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,**{k:v for k,v in DECODING.items() if k!='context_limit'},
                    use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            out={**{k:job[k] for k in ('id','condition','seed','prompt_sha256')},'output':text,'synthetic':False,
                'hit_output_limit':bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before}
            outputs.append(out);write_lines(output/'generations.private.jsonl',outputs)
            if not text or out['hit_output_limit']:raise RuntimeError('Empty or cutoff output; incomplete run')
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
        sources={r['id']:r['source'] for r in pack['rows']};semantic={};verdict_rows=[]
        for i,out in enumerate(outputs):
            phase('diagnostics_'+str(i+1)+'_of_24')
            verdict=verifier.verify(sources[out['id']],out['output'],predict)
            semantic[(out['id'],out['condition'],out['seed'])]=verdict
            verdict_rows.append({**{k:out[k] for k in ('id','condition','seed')},'source_sha256':sha(sources[out['id']]),'output_sha256':sha(out['output']),'verifier':verdict})
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
            status['completed_verifications']=len(verdict_rows)
        records,summary=evaluate(pack,outputs,semantic)
        write_lines(output/'jobs.private.jsonl',jobs);write_lines(output/'metrics.private.jsonl',records);write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_model':NLI_MODEL,'nli_revision':NLI_REVISION,
            'decoding':DECODING,'elapsed_seconds':time.monotonic()-started,'generation_memory':generation_memory,
            'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True);phase('complete')
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',exception_type=type(exc).__name__);write_json(output/'run_status.json',status)
        print('Inference failed; exception details suppressed to protect credentials.',flush=True)
        raise RuntimeError('Inference incomplete; inspect saved status and partial artifacts') from None


def synthetic_pack():
    old=previous.synthetic_pack()
    return {k:old[k] for k in ('rows','style_profile','policy')}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument('--prepare-pack',action='store_true');a.add_argument('--dry-run',action='store_true');a.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.dry_run:
        pack=synthetic_pack();jobs=jobs_for(pack);rows={r['id']:r for r in pack['rows']}
        outputs=[{**{k:j[k] for k in ('id','condition','seed','prompt_sha256')},'output':rows[j['id']]['source'],'synthetic':True,'hit_output_limit':False} for j in jobs]
        _,summary=evaluate(pack,outputs,synthetic=True);fresh(args.output_dir);write_json(args.output_dir/'synthetic_evaluation.json',summary)
        print('Synthetic CPU check: 24 jobs; zero model calls or training.');return
    if args.pack is None:p.error('--pack required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'status':'preflight_passed','jobs':len(jobs_for(pack)),'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret access, model download or generation.')


if __name__=='__main__':main()
