"""Realize source groups separately and assemble verbatim; inference only."""
import argparse
from collections import Counter
import copy
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import bound_grouping as prior
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,tokens
from fama_style.research.signal import features
from fama_style.research.style_drift import score_candidate

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='groupwise_realization_v1'
SEEDS=prior.SEEDS
DECODING=prior.DECODING
CONDITIONS=('A_whole_paragraph','B_groupwise')


def budgets(row,total=768):
    """Largest-remainder token allocation, proportional to source-group word counts."""
    units={u['unit_id']:u['text'] for u in row['source_units']}
    weights=[max(1,len(tokens(' '.join(units[i] for i in group)))) for group in row['groups']]
    if not weights or total<len(weights):raise ValueError('Insufficient output budget')
    # Reserve one token per group; distribute the rest deterministically.
    remaining=total-len(weights);denominator=sum(weights)
    allocated=[1+remaining*w//denominator for w in weights]
    order=sorted(range(len(weights)),key=lambda i:(-(remaining*weights[i]%denominator),i))
    for i in order[:total-sum(allocated)]:allocated[i]+=1
    return allocated


def jobs_for(pack):
    # Reuse the old prompt constructor, source-span checks and oracle grouping exactly.
    old=prior.jobs_for(pack);controls={(j['id'],j['seed']):j for j in old if j['condition']=='C_oracle_groups'}
    result=[]
    for index,row in enumerate(pack['rows']):
        allocation=budgets(row,DECODING['max_new_tokens'])
        for seed_index,seed in enumerate(SEEDS):
            base=controls[(row['id'],seed)]
            a={**base,'condition':'A_whole_paragraph','group_index':None,'sampling_seed':seed,'max_new_tokens':768}
            b=[]
            for i,(group,limit) in enumerate(zip(row['groups'],allocation),1):
                prompt=(base['prompt']+'\n\nThis call realizes only source group '+str(i)+': '+json.dumps(group)+'. '
                    'Write exactly one sentence expressing this group. The full source and other groups are context only: '
                    'use them to resolve references, and do not move their claims into this sentence. '
                    'Keep the same style and semantic tolerance. Return only this sentence, without labels or commentary. '
                    'Do not generate sentences for the other groups.')
                sampling_seed=int(sha(PROTOCOL+'|'+row['id']+'|'+str(seed)+'|'+str(i))[:8],16)%(2**31)
                b.append({'id':row['id'],'condition':'B_groupwise','seed':seed,'group_index':i,
                    'sampling_seed':sampling_seed,'max_new_tokens':limit,'prompt':prompt,
                    'prompt_sha256':sha(prompt),'source_sha256':row['source_sha256'],'oracle':True})
            result.extend([a]+b if (index+seed_index)%2==0 else b+[a])
    return result


def job_key(row):return (row['id'],row['condition'],row['seed'],row['group_index'])
def candidate_key(row):return (row['id'],row['condition'],row['seed'])


def piece_format(text):
    parsed=prior.output_units(text)
    return {'single_sentence':len(parsed['units'])==1 and not parsed['ambiguous'],
        'boundary_ambiguous':parsed['ambiguous'],'sentence_count':len(parsed['units']),
        'paragraph_format':prior.parent.paragraph_format(text),
        'terminal_punctuation':bool(re.search(r'[.!?][)\]”’"\']*$',text.strip()))}


def assemble(pack,outputs,synthetic=False):
    expected={job_key(j):j for j in jobs_for(pack)};seen=set();indexed={}
    if len(outputs)!=len(expected):raise ValueError('Incomplete generation calls')
    for out in outputs:
        key=job_key(out);job=expected.get(key)
        if job is None or key in seen:raise ValueError('Unknown or duplicate call')
        if (out['prompt_sha256']!=job['prompt_sha256'] or out['max_new_tokens']!=job['max_new_tokens']
            or out['sampling_seed']!=job['sampling_seed']):raise ValueError('Changed prompt, budget or sampling seed')
        if not out['output'].strip() or out['hit_output_limit'] is not False or out.get('synthetic') is not synthetic:
            raise ValueError('Empty, cutoff or mislabeled generation')
        seen.add(key);indexed[key]=out
    candidates=[]
    for row in pack['rows']:
        for seed in SEEDS:
            for condition in CONDITIONS:
                pieces=[indexed[(row['id'],condition,seed,None)]] if condition=='A_whole_paragraph' else [indexed[(row['id'],condition,seed,i)] for i in range(1,len(row['groups'])+1)]
                # Join returned strings in group order, without trimming, repairs or fallback.
                text=' '.join(p['output'] for p in pieces)
                formats=[piece_format(p['output']) for p in pieces]
                candidates.append({'id':row['id'],'condition':condition,'seed':seed,'paragraph':text,'output_sha256':sha(text),
                    'piece_output_sha256':[sha(p['output']) for p in pieces],
                    'group_outputs':[p['output'] for p in pieces] if condition=='B_groupwise' else None,
                    'group_piece_formats':formats if condition=='B_groupwise' else None,
                    'assembly_format_valid':all(f['single_sentence'] and f['paragraph_format'] and f['terminal_punctuation'] for f in formats) if condition=='B_groupwise' else None,
                    'generation_calls':len(pieces),'max_output_token_allowance':sum(p['max_new_tokens'] for p in pieces),
                    **{k:sum(p.get(k,0) for p in pieces) for k in ('input_tokens','output_tokens','generation_seconds')}})
    return candidates


def binding_diagnostics(row,candidate,predict=None):
    if candidate['condition']=='A_whole_paragraph':
        return {'method':'positional_only_when_counts_match',**prior.group_diagnostics(row,'C_oracle_groups',candidate['paragraph'],predict)}
    units={u['unit_id']:u['text'] for u in row['source_units']};checks=[]
    for i,(group,text) in enumerate(zip(row['groups'],candidate['group_outputs']),1):
        source=' '.join(units[n] for n in group)
        checks.append({'group_index':i,'source_unit_ids':group,'source_group_sha256':sha(source),
            'output_piece_sha256':sha(text),'format':piece_format(text),'verifier':verifier.verify(source,text,predict) if predict else None})
    parsed=prior.output_units(candidate['paragraph'])
    return {'method':'known_generation_call_assignment','requested_groups':row['groups'],
        'output_sentence_count':len(parsed['units']),'segmentation_ambiguous':parsed['ambiguous'],
        'count_adherence':None if parsed['ambiguous'] else len(parsed['units'])==len(row['groups']),
        'assembly_format_valid':candidate['assembly_format_valid'],'assigned_group_checks':checks,'group_binding_verified':False}


def evaluate(pack,outputs,semantic=None,binding=None,synthetic=False):
    candidates=assemble(pack,outputs,synthetic);rows={r['id']:r for r in pack['rows']};keys={candidate_key(c) for c in candidates}
    if not synthetic and (set(semantic or {})!=keys or set(binding or {})!=keys):raise ValueError('Missing paragraph or group evidence')
    records=[]
    for c in candidates:
        key=candidate_key(c);row=rows[c['id']];verdict=(semantic or {}).get(key);basic=binding_diagnostics(row,c)
        supplied=(binding or {}).get(key,basic)
        for field in ('method','requested_groups','output_sentence_count','segmentation_ambiguous','count_adherence'):
            if supplied[field]!=basic[field]:raise ValueError('Group diagnostics do not match assembly')
        checks=supplied.get('assigned_group_checks',supplied.get('positional_group_checks',[]))
        if c['condition']=='B_groupwise':
            if supplied['assembly_format_valid']!=basic['assembly_format_valid']:raise ValueError('Changed assembly validity')
            if len(checks)!=len(row['groups']):raise ValueError('Missing per-call group checks')
            for actual,expected in zip(checks,basic['assigned_group_checks']):
                for field in ('group_index','source_unit_ids','source_group_sha256','output_piece_sha256','format'):
                    if actual[field]!=expected[field]:raise ValueError('Group verdict not bound to generated piece')
                if not synthetic and actual['verifier'] is None:raise ValueError('Missing group verdict')
        elif checks and basic['count_adherence'] is not True:
            raise ValueError('Positional checks without resolved sentence alignment')
        elif not synthetic and basic['count_adherence'] is True:
            if len(checks)!=len(row['groups']):raise ValueError('Missing whole-paragraph positional checks')
            units={u['unit_id']:u['text'] for u in row['source_units']};parts=prior.output_units(c['paragraph'])['units']
            for i,(group,check) in enumerate(zip(row['groups'],checks),1):
                if (check['source_unit_ids']!=group or check['output_position']!=i
                    or check['source_group_sha256']!=sha(' '.join(units[n] for n in group))
                    or check['output_sentence_sha256']!=sha(parts[i-1])):raise ValueError('Positional verdict hash mismatch')
        records.append({**c,'candidate_id':sha(row['id']+'|'+c['paragraph'])[:20],'verifier':verdict,'group_diagnostics':supplied,
            'format_valid':prior.parent.paragraph_format(c['paragraph']),
            'output_features':features(c['paragraph']),
            'historical_control_output_matches':next((r['output_sha256']==c['output_sha256'] for r in pack.get('historical_controls',[]) if r['id']==c['id'] and r['seed']==c['seed']),None) if c['condition']=='A_whole_paragraph' else None,
            **score_candidate(row['source'],c['paragraph'],pack['style_profile'],verdict,pack['policy'])})
    conditions={}
    for condition in CONDITIONS:
        group=[r for r in records if r['condition']==condition]
        statuses=[check['verifier']['status'] for r in group for check in r['group_diagnostics'].get('assigned_group_checks',r['group_diagnostics'].get('positional_group_checks',[])) if check['verifier']]
        conditions[condition]={'paragraphs':len(group),'generation_calls':sum(r['generation_calls'] for r in group),
            'valid_paragraph_formats':sum(r['format_valid'] for r in group),
            'count_matches':sum(r['group_diagnostics']['count_adherence'] is True for r in group),
            'valid_assemblies':sum(r['assembly_format_valid'] is True for r in group) if condition=='B_groupwise' else None,
            'mean_style_benefit_proxy':statistics.mean(r['style_alignment_benefit_proxy'] for r in group),
            'paragraph_verifier_statuses':dict(Counter(r['old_verifier_status'] for r in group)),
            'group_verifier_statuses':dict(Counter(statuses)),
            'historical_control_output_matches':sum(r['historical_control_output_matches'] is True for r in group) if condition=='A_whole_paragraph' else None,
            **{k:sum(r[k] for r in group) for k in ('input_tokens','output_tokens','generation_seconds')}}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete','conditions':conditions,
        'training_called':False,'reserved_test_rows_read':0,'validation_rows_read':0,'semantic_accuracy':None,'author_style_accuracy':None,'qualitative_assessment':'pending',
        'limitations':'Three selected training pairs, oracle plans, nine candidates per strategy. Group-call assignment/assembly is guaranteed by code; semantic coverage, one-sentence realization and coherence are not. Per-call input overhead is unequal; output allowances sum to 768 per paragraph. No best-of-three quality claim.'}


def runtime_paths():return prior.runtime_paths()


def prepare_pack(output):
    root=ROOT/'research_runs/bound_grouping_realization_v1';release=json.loads((root/'release_verification.json').read_text())
    old=prior.load_pack(root/'input_pack/bound_grouping_pack.json',release['pack_sha256'])
    review=json.loads((root/'version_1_review/verification.json').read_text());run=root/'kaggle_version_1/bound_grouping_realization_v1'
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Prior run review failed')
    for name,h in review['artifact_hashes'].items():
        if file_sha(run/name)!=h:raise ValueError('Prior result changed')
    controls=[r for r in read_lines(run/'generations.private.jsonl') if r['condition']=='C_oracle_groups']
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'train_sha256':TRAIN_SHA,
        'rows':copy.deepcopy(old['rows']),'style_profile':old['style_profile'],'policy':old['policy'],'decoding':DECODING,
        'parent_pack_sha256':release['pack_sha256'],'historical_controls':[{'id':r['id'],'seed':r['seed'],'prompt_sha256':r['prompt_sha256'],'output_sha256':sha(r['output'])} for r in controls],
        'scope':'Known training-example oracle grouping, whole paragraph versus group calls plus verbatim assembly. Full source context visible in all calls. No generated context is passed to later groups.',
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}}
    expected={(r['id'],r['seed']):r for r in pack['historical_controls']}
    if len(expected)!=9 or any(j['prompt_sha256']!=expected[(j['id'],j['seed'])]['prompt_sha256'] for j in jobs_for(pack) if j['condition']=='A_whole_paragraph'):raise ValueError('Historical prompt controls changed')
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))
    fresh(output);write_json(output/'groupwise_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':3,'paragraph_candidates':18,'generation_calls':len(jobs_for(pack)),'pack_sha256':file_sha(output/'groupwise_pack.json'),'training_called':False})
    print('Prepared 9 whole-paragraph calls + 33 group calls = 18 paragraph candidates. No inference executed.');return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if (pack['protocol']!=PROTOCOL or pack['model_id']!=MODEL_ID or pack['model_revision']!=MODEL_REVISION
        or pack['train_sha256']!=TRAIN_SHA or pack['decoding']!=DECODING):raise ValueError('Changed experiment identity')
    if pack['policy']!=json.loads((ROOT/prior.POLICY).read_text()):raise ValueError('Changed grading policy')
    for row in pack['rows']:prior.validate_grouping(row['source'],row['reference'],row['groups'],row['source_sha256'],row['reference_sha256'])
    if pack['style_profile']['fit_split']!='train' or pack['style_profile']['training_rows']!=247:raise ValueError('Wrong profile split')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Changed jobs')
    return pack


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256'] or not ids or len(ids)+job['max_new_tokens']>DECODING['context_limit']:raise ValueError('Changed/over-context prompt; no truncation')
    return {'id':job['id'],'condition':job['condition'],'seed':job['seed'],'group_index':job['group_index'],
        'input_tokens':len(ids),'max_new_tokens':job['max_new_tokens'],'input_ids_sha256':sha(json.dumps(ids))}


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
            outputs.append(out);write_lines(output/'generations.private.jsonl',outputs)
            if not text or out['hit_output_limit']:raise RuntimeError('Empty or cutoff output; incomplete run')
        generation_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize;gc.collect();torch.cuda.empty_cache()
        candidates=assemble(pack,outputs);write_lines(output/'paragraphs.private.jsonl',candidates)
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
        rows={r['id']:r for r in pack['rows']};semantic={};verdict_rows=[];binding={};group_rows=[]
        for i,candidate in enumerate(candidates):
            phase('diagnostics_'+str(i+1)+'_of_'+str(len(candidates)))
            row=rows[candidate['id']];key=candidate_key(candidate)
            verdict=verifier.verify(row['source'],candidate['paragraph'],predict);semantic[key]=verdict
            verdict_rows.append({**{k:candidate[k] for k in ('id','condition','seed')},'source_sha256':row['source_sha256'],
                'output_sha256':candidate['output_sha256'],'verifier':verdict})
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
            diagnostic=binding_diagnostics(row,candidate,predict);binding[key]=diagnostic
            group_rows.append({**{k:candidate[k] for k in ('id','condition','seed')},'output_sha256':candidate['output_sha256'],'group_diagnostics':diagnostic})
            write_lines(output/'group_verifier_results.private.jsonl',group_rows)
            status['completed_paragraph_verifications']=len(verdict_rows)
        records,summary=evaluate(pack,outputs,semantic,binding)
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
    old=prior.synthetic_pack()
    return {k:old[k] for k in ('rows','style_profile','policy')}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group()
    actions.add_argument('--prepare-pack',action='store_true');actions.add_argument('--dry-run',action='store_true');actions.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.dry_run:
        pack=synthetic_pack();rows={r['id']:r for r in pack['rows']};outputs=[]
        for j in jobs_for(pack):
            row=rows[j['id']];units={u['unit_id']:u['text'] for u in row['source_units']}
            text=row['source'] if j['condition']=='A_whole_paragraph' else ' '.join(units[i] for i in row['groups'][j['group_index']-1])
            outputs.append({**{k:j[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},
                'output':text,'synthetic':True,'hit_output_limit':False})
        _,summary=evaluate(pack,outputs,synthetic=True);fresh(args.output_dir);write_json(args.output_dir/'synthetic_evaluation.json',summary)
        print('Synthetic groupwise wiring check; no model or Secret accessed.');return
    if args.pack is None:p.error('--pack required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'status':'preflight_passed','generation_calls':len(jobs_for(pack)),'paragraph_candidates':18,'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret, weights or GPU accessed.')


if __name__=='__main__':main()
