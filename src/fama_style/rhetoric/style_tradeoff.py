"""Small matched candidate comparison under a soft style/drift objective; no training."""
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
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,tokens
from fama_style.research.signal import fit_profile,features
from fama_style.research.style_drift import score_candidate,frontier

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='style_tradeoff_candidates_v1'
SEEDS=(41,42,43)
CONDITIONS=('A_direct','B_reference_operations')
DECODING={'do_sample':True,'temperature':0.6,'top_p':0.9,'top_k':0,'max_new_tokens':768,'context_limit':4096}
POLICY=Path('docs/style_drift_tradeoff_v1/policy.json')
PATTERNS={'explicit_contrast':r'\b(?:however|although|though|whereas|in contrast|on the other hand|but)\b',
          'author_voice':r'\b(?:we|our)\b'}
INSTRUCTION=('Rewrite the passage in the academic economics style of Eugene Fama and his coauthors. '
    'Keep the economic argument and main claims intact. Small changes in emphasis or secondary wording are acceptable '
    'when they improve the writing. Do not invent findings, numbers, actors or causal mechanisms. '
    'Sentence count is free; longer sentences and inflated vocabulary are not goals. '
    'Return only one coherent paragraph without commentary or Markdown.')


def empirical_prior(train):
    if len(train)!=247 or any(r['split']!='train' for r in train):raise ValueError('Expected frozen training split')
    return {'fit_split':'train','training_rows':247,'features':{
        k:{'source_present':sum(bool(re.search(p,r['plain_english'],re.I)) for r in train),
           'reference_present':sum(bool(re.search(p,r['completion'],re.I)) for r in train)}
        for k,p in PATTERNS.items()},
        'rule':'Eligible when reference presence exceeds source by at least 5/247 and reference prevalence >= 25%; apply only to an existing source relation or voice.',
        'limitations':'Two shallow training-derived operations, not a learned rhetorical taxonomy or an author classifier.'}


def operations(source,prior):
    result=[]
    for k,counts in prior['features'].items():
        if counts['reference_present']<.25*247 or counts['reference_present']-counts['source_present']<5:continue
        if not re.search(PATTERNS[k],source,re.I):continue
        result.append({'operation':k,'instruction':{
            'explicit_contrast':'Make an existing contrast explicit at a clause or sentence boundary; do not create a new contrast or causal link.',
            'author_voice':'Retain the existing we/our research voice for methods or findings; avoid replacing it with impersonal nominal phrases.'}[k]})
    return result


def jobs_for(pack):
    rows=pack['rows']
    if len(rows)!=4 or len({r['id'] for r in rows})!=4 or any(r['split']!='validation' or not r['source'].strip() for r in rows):
        raise ValueError('Expected four unique development inputs')
    result=[]
    for index,r in enumerate(rows):
        base=INSTRUCTION+'\n\nPassage:\n'+r['source']
        plan=operations(r['source'],pack['empirical_prior'])
        guided=base+'\n\nOptional reference-derived operations:\n'+json.dumps(plan,ensure_ascii=False,separators=(',',':'))
        for seed in SEEDS:
            for condition in CONDITIONS if index%2==0 else CONDITIONS[::-1]:
                prompt=base if condition=='A_direct' else guided
                result.append({'id':r['id'],'condition':condition,'seed':seed,'prompt':prompt,
                    'prompt_sha256':sha(prompt),'source_sha256':sha(r['source']),'oracle':False})
    return result


def runtime_paths():
    return parent.runtime_paths()+[ROOT/POLICY]


def prepare_pack(output):
    trainpath=ROOT/'data/fama_corpus_v1/train.jsonl';valpath=ROOT/'data/fama_corpus_v1/validation.jsonl'
    if file_sha(trainpath)!=TRAIN_SHA or file_sha(valpath)!=parent.VALIDATION_SHA:raise ValueError('Frozen corpus changed')
    train,val=read_lines(trainpath),read_lines(valpath)
    if len(val)!=21 or any(r['split']!='validation' for r in val):raise ValueError('Wrong development split')
    ranked=sorted(val,key=lambda r:(len(tokens(r['plain_english'])),r['id']))
    # Median rank of each of four consecutive length strata; no outcomes consulted.
    indices=[(i*len(ranked)//4+((i+1)*len(ranked)//4)-1)//2 for i in range(4)]
    selected=[ranked[i] for i in indices]
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'train_sha256':TRAIN_SHA,'validation_sha256':parent.VALIDATION_SHA,
        'rows':[{'id':r['id'],'split':'validation','source':r['plain_english'],'reference':r['completion']} for r in selected],
        'style_profile':fit_profile(train),'empirical_prior':empirical_prior(train),
        'policy':json.loads((ROOT/POLICY).read_text()),'decoding':DECODING,
        'selection':{'rule':'Lower median rank in each of four source-token-length strata; ties by ID; no output scores used.',
                     'ranks_zero_based':indices,'previously_exposed_development':True,'reserved_test_rows_read':0},
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}}
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))
    fresh(output);write_json(output/'style_tradeoff_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':4,'jobs':24,
        'pack_sha256':file_sha(output/'style_tradeoff_pack.json'),'training_called':False,'reserved_test_rows_read':0})
    print('Prepared four development cases / 24 inference jobs; no models accessed.');return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if (pack['protocol']!=PROTOCOL or pack['model_id']!=MODEL_ID or pack['model_revision']!=MODEL_REVISION
        or pack['train_sha256']!=TRAIN_SHA or pack['validation_sha256']!=parent.VALIDATION_SHA
        or pack['decoding']!=DECODING):raise ValueError('Changed experiment identity')
    if pack['policy']!=json.loads((ROOT/POLICY).read_text()):raise ValueError('Changed scoring policy')
    if (pack['style_profile']['fit_split']!='train' or pack['empirical_prior']['fit_split']!='train'
        or pack['style_profile']['training_rows']!=247 or pack['empirical_prior']['training_rows']!=247):raise ValueError('Non-training prior')
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
            'output_features':features(out['output']),**scores})
    # Do not select winners using references or discard semantic warnings. Show all candidates.
    fronts={r['id']:frontier([{'candidate_id':'source-control','style_alignment_benefit_proxy':0,'semantic_cost_proxy':0}]+
        [m for m in records if m['id']==r['id'] and m['format_valid']]) for r in pack['rows']}
    counts={c:{'outputs':sum(r['condition']==c for r in records),
        'unique_paragraphs':len({r['output_sha256'] for r in records if r['condition']==c}),
        'mean_style_benefit_proxy':statistics.mean(r['style_alignment_benefit_proxy'] for r in records if r['condition']==c),
        'verifier_statuses':dict(Counter(r['old_verifier_status'] for r in records if r['condition']==c))} for c in CONDITIONS}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete',
        'conditions':counts,'within_case_frontiers':fronts,'training_called':False,'reserved_test_rows_used':0,
        'qualitative_style_assessment':'pending','semantic_accuracy':None,'author_style_accuracy':None,
        'limitations':'Four previously exposed development cases; sampling diversity is not independent replication. No best-of-three claim or qualitative style result before assessment. Proxies are not calibrated severity or author-style measures.'}


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
        sources={r['id']:r['source'] for r in pack['rows']};semantic={}
        for i,out in enumerate(outputs):
            phase('diagnostics_'+str(i+1)+'_of_24')
            semantic[(out['id'],out['condition'],out['seed'])]=verifier.verify(sources[out['id']],out['output'],predict)
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
    train=[{'split':'train','plain_english':'Firms invest.','completion':'Firms invest.'} for _ in range(247)]
    return {'rows':[{'id':'synthetic-'+str(i),'split':'validation','source':'We invest, but banks may lend.','reference':'We invest; banks may lend.'} for i in range(4)],
        'empirical_prior':empirical_prior(train),'style_profile':fit_profile(train),'policy':json.loads((ROOT/POLICY).read_text())}


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
