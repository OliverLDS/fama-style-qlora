"""Frozen v3: scoped endpoint/causality guards over the unchanged v2 verifier."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from . import claim_verifier as base
from .relation_guards import check
from .claims import sha,sentences
from fama_style.rhetoric.semantics import checked_prediction,NLI_MODEL

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='claim_verifier_calibration_v3'
NLI_REVISION=base.NLI_REVISION


def verify(source,output,predict):
    previous=base.verify(source,output,predict)
    evidence=check(source,output);status=previous['status']
    if evidence['conflicts']:status='risk_signal'
    elif evidence['unresolved'] and status=='model_consistent':status='abstain'
    return {**previous,'v2_status':previous['status'],'status':status,'scoped_relation_guards':evidence}


def replay_v2(records):
    """Reuse saved probabilities as development regressions, never fresh evidence."""
    results=[]
    for row in records:
        old=row['verdict'];source,output=row['source'],row['output'];cache={}
        def add(a,b,p):
            key=(sha(a),sha(b))
            if key in cache and cache[key]!=p:raise ValueError('Inconsistent cached probabilities')
            cache[key]=p
        pairs=[(source,output),(output,source)]+[(output,s) for s in sentences(source)]+[(source,s) for s in sentences(output)]
        if len(pairs)!=len(old['global_nli']['checks']):raise ValueError('Incomplete global probabilities')
        for (a,b),r in zip(pairs,old['global_nli']['checks']):add(a,b,r['probabilities'])
        for alignment in old['alignments']:
            evidence=old['output_units'] if alignment['direction']=='source_coverage' else old['source_units']
            candidates={tuple(b['unit_ids']):b['text'] for b in base.bundles(evidence)}
            for c in alignment['candidates']:add(candidates[tuple(c['evidence_unit_ids'])],alignment['claim_text'],c['probabilities'])
        result=verify(source,output,lambda a,b:cache[(sha(a),sha(b))])
        if result['v2_status']!=old['status']:raise ValueError('V2 baseline changed')
        results.append({**row,'original_v2_split':row['split'],'split':'development','verdict':result})
    return results


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(folder):
    p=json.loads((folder/'protocol.json').read_text())
    if p['protocol']!=PROTOCOL:raise ValueError('Wrong calibration protocol')
    for name,h in p['code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen verifier source changed; create a new calibration version')
    for name,h in p['fixture_files'].items():
        if file_sha(folder/name)!=h:raise ValueError('Frozen synthetic cases changed')
    cases=[]
    for phase in ('development','held_out'):
        group=json.loads((folder/(phase+'.json')).read_text())
        if len(group)!=p['expected_counts'][phase] or any(c['split']!=phase for c in group):raise ValueError('Wrong split size or label')
        if any(c['expected'] not in ('faithful','changed') or c['variant'] not in ('identity','paraphrase','mutation') for c in group):raise ValueError('Invalid case labels')
        cases+=group
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    for key in ('source','scenario_group'):
        if {c[key] for c in cases if c['split']=='development'}&{c[key] for c in cases if c['split']=='held_out'}:raise ValueError('Calibration split leakage')
    return p,cases


def score(records):
    good=[r for r in records if r['expected']=='faithful'];bad=[r for r in records if r['expected']=='changed'];hard=[r for r in good if r['variant']=='paraphrase']
    rate=lambda rows,status:sum(r['verdict']['status']==status for r in rows)/len(rows) if rows else None
    return {'rows':len(records),'faithful_rows':len(good),'changed_rows':len(bad),'error_detection_rate':rate(bad,'risk_signal'),
        'faithful_false_alarm_rate':rate(good,'risk_signal'),'nontrivial_faithful_acceptance_rate':rate(hard,'model_consistent'),
        'changed_false_acceptance_rate':rate(bad,'model_consistent'),'abstention_rate':rate(records,'abstain'),
        'status_counts':dict(Counter(r['verdict']['status'] for r in records))}


def gate(metrics,criteria):
    checks={
        'error_detection':metrics['error_detection_rate'] is not None and metrics['error_detection_rate']>=criteria['minimum_error_detection_rate'],
        'faithful_false_alarms':metrics['faithful_false_alarm_rate'] is not None and metrics['faithful_false_alarm_rate']<=criteria['maximum_faithful_false_alarm_rate'],
        'paraphrase_acceptance':metrics['nontrivial_faithful_acceptance_rate'] is not None and metrics['nontrivial_faithful_acceptance_rate']>=criteria['minimum_nontrivial_faithful_acceptance_rate'],
        'changed_false_acceptance':metrics['changed_false_acceptance_rate'] is not None and metrics['changed_false_acceptance_rate']<=criteria['maximum_changed_false_acceptance_rate']}
    return {'passed':all(checks.values()),'checks':checks}


def evaluate(folder,output,predict,execution_kind):
    if output.exists():raise ValueError('Use a fresh output directory')
    p,cases=validate_protocol(folder);output.mkdir(parents=True);records=[];started=time.monotonic()
    (output/'run_status.json').write_text(json.dumps({'complete':False,'completed_cases':0,'total_cases':len(cases),'training_called':False})+'\n')
    for i,c in enumerate(cases,1):
        records.append({**c,'verdict':verify(c['source'],c['output'],predict)})
        (output/'partial_results.json').write_text(json.dumps(records,indent=2)+'\n')
        (output/'run_status.json').write_text(json.dumps({'complete':False,'completed_cases':i,'total_cases':len(cases),'training_called':False})+'\n')
        print('Assessed synthetic case '+str(i)+'/'+str(len(cases)),flush=True)
    summary={'protocol':PROTOCOL,'execution_kind':execution_kind,'training_called':False,'gemma_generation_called':False,
        'private_dataset_rows_used':0,'splits':{s:score([r for r in records if r['split']==s]) for s in ('development','held_out')},
        'held_out_by_category':{k:score([r for r in records if r['split']=='held_out' and r['category']==k]) for k in sorted({r['category'] for r in records if r['split']=='held_out'})},
        'elapsed_seconds':time.monotonic()-started,'semantic_proof':False,
        'limitations':'Authored synthetic controls, not blinded or independently annotated domain accuracy. All previously exposed v1/v2 examples are development-only; only fresh v3 cases are held out.'}
    summary['readiness_gate']=gate(summary['splits']['held_out'],p['readiness_gate']) if execution_kind=='pinned_nli_inference' else {'passed':False,'reason':'Synthetic stub execution cannot validate a verifier'}
    baseline=[{**r,'verdict':{'status':{'proxy_consistent':'model_consistent','proxy_flagged':'risk_signal'}.get(r['verdict'].get('global_nli',{}).get('status'),'abstain')}} for r in records]
    summary['baseline_global_sentence_nli']={s:score([r for r in baseline if r['split']==s]) for s in ('development','held_out')}
    summary['held_out_baseline_passes_overridden']=sum(r['split']=='held_out' and r['verdict'].get('global_nli',{}).get('status')=='proxy_consistent' and r['verdict']['status']=='risk_signal' for r in records)
    v2_baseline=[{**r,'verdict':{'status':r['verdict']['v2_status']}} for r in records]
    summary['baseline_v2_claim_verifier']={s:score([r for r in v2_baseline if r['split']==s]) for s in ('development','held_out')}
    summary['held_out_v2_passes_overridden']=sum(r['split']=='held_out' and r['verdict']['v2_status']=='model_consistent' and r['verdict']['status']=='risk_signal' for r in records)
    summary['held_out_v2_passes_abstained']=sum(r['split']=='held_out' and r['verdict']['v2_status']=='model_consistent' and r['verdict']['status']=='abstain' for r in records)
    validate_protocol(folder)
    (output/'results.json').write_text(json.dumps(records,indent=2)+'\n');(output/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'run_status.json').write_text(json.dumps({'complete':True,'completed_cases':len(cases),'total_cases':len(cases),'training_called':False})+'\n')
    (output/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'protocol_sha256':file_sha(folder/'protocol.json'),
        'code_files':p['code_files'],'fixture_files':p['fixture_files'],
        'files':{n:file_sha(output/n) for n in ('results.json','evaluation.json','run_status.json')}},indent=2)+'\n')
    return summary


def run_training(folder,output):
    """Pretrained NLI calibration only; lazy ML imports, no optimizer or trainer."""
    validate_protocol(folder)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from transformers.utils import logging
    from fama_style.rhetoric.experiment import with_heartbeat
    logging.set_verbosity_error()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Select T4 x2 and expose GPU 0 before importing torch')
    torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats(0)
    print('Loading pinned public NLI model; no Secret required.',flush=True)
    tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer load')
    model=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False,dtype=torch.float32).to('cuda:0'),'NLI model load')
    model.eval();model.requires_grad_(False);labels={int(k):v.lower() for k,v in model.config.id2label.items()}
    if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Unexpected NLI labels')
    limit=min(512,int(tokenizer.model_max_length),int(getattr(model.config,'max_position_embeddings',512)));cache={}
    def predict(premise,hypothesis):
        key=(sha(premise),sha(hypothesis))
        if key in cache:return cache[key]
        batch=tokenizer(premise,hypothesis,truncation=False,return_tensors='pt')
        if int(batch['input_ids'].shape[-1])>limit:cache[key]=None;return None
        with torch.inference_mode():
            values=torch.softmax(model(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
        cache[key]={labels[i]:float(v) for i,v in enumerate(values)};return cache[key]
    summary=evaluate(folder,output,predict,'pinned_nli_inference')
    runtime={'model_id':NLI_MODEL,'model_revision':NLI_REVISION,'dtype':'float32','device':torch.cuda.get_device_name(0),'context_limit':limit,
        'cache_entries':len(cache),'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
        'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','huggingface-hub')},'training_called':False}
    (output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    manifest=json.loads((output/'manifest.json').read_text());manifest['files']['runtime.json']=file_sha(output/'runtime.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'readiness_gate':summary['readiness_gate'],'splits':summary['splits']},indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'docs/claim_verifier_calibration_v3')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--execute-calibration',action='store_true')
    a=p.parse_args(argv)
    if a.execute_calibration:run_training(a.protocol_dir,a.output_dir)
    else:
        protocol,cases=validate_protocol(a.protocol_dir)
        if a.output_dir.exists():raise ValueError('Use a fresh output directory')
        a.output_dir.mkdir(parents=True)
        (a.output_dir/'preflight.json').write_text(json.dumps({'protocol':PROTOCOL,'cases':len(cases),'model_calls':0,'training_called':False,'execution_kind':'preflight_only'},indent=2)+'\n')
        print('Preflight passed. No model download, Secret access, generation or training.')


if __name__=='__main__':main()
