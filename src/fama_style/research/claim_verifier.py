"""Claim-unit NLI and bounded role guards. Calibration only; no generation/training."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

from .claim_audit import claim_checks,NUMBER_WORDS
from .claims import sentences,sha
from fama_style.rhetoric.semantics import assess,checked_prediction,NLI_MODEL

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='claim_verifier_calibration_v2'
NLI_REVISION='6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
MAX_UNITS=12
MAX_PAIRS=256
NUMBER=r'(?:\d+(?:\.\d+)?|'+ '|'.join(NUMBER_WORDS)+')'
ACTOR=r'(?:the\s+)?[a-z]+(?:\s+[a-z]+){0,3}?'
Q_ACTIVE=re.compile(r'(?P<actor>'+ACTOR+r')\s+(?:owns?|holds?)\s+(?P<number>'+NUMBER+r')\s+(?P<unit>shares?|bonds?|notes?|units?)',re.I)
Q_PASSIVE=re.compile(r'(?P<number>'+NUMBER+r')\s+(?P<unit>shares?|bonds?|notes?|units?)\s+(?:are|is)\s+(?:owned|held)\s+by\s+(?P<actor>'+ACTOR+r')',re.I)
PAY_ACTIVE=re.compile(r'(?P<actor>'+ACTOR+r')\s+(?P<verb>pays?|reimburses?)\s+(?P<recipient>'+ACTOR+r')',re.I)
PAY_PASSIVE=re.compile(r'(?P<recipient>'+ACTOR+r')\s+(?:is|are)\s+(?P<verb>paid|reimbursed)\s+by\s+(?P<actor>'+ACTOR+r')',re.I)


def clean_actor(value):return re.sub(r'^the\s+','',value.lower().strip())
def atom(value):return format(Decimal(str(NUMBER_WORDS.get(value.lower(),value))).normalize(),'f')


def explicit_actor(actor):
    blocked={'it','they','he','she','we','may','might','could','must','should','will','would','not','no','all','every','some','many','most','if','unless','when','because','that','which'}
    return bool(actor) and not (set(actor.split())&blocked)


def role_fact(unit):
    body=unit.strip().rstrip('.!?').strip()
    for pattern in (Q_ACTIVE,Q_PASSIVE):
        m=pattern.fullmatch(body)
        if m:
            actor=clean_actor(m['actor'])
            if not explicit_actor(actor):return None
            return {'relation':'holding','actor':actor,'unit':m['unit'].lower().rstrip('s'),'value':atom(m['number'])}
    for pattern in (PAY_ACTIVE,PAY_PASSIVE):
        m=pattern.fullmatch(body)
        if m:
            actor,recipient=clean_actor(m['actor']),clean_actor(m['recipient'])
            if not explicit_actor(actor) or not explicit_actor(recipient):return None
            return {'relation':'reimburse' if m['verb'].lower().startswith('reimburs') else 'pay','actor':actor,'recipient':recipient}
    return None


def units(text):
    """Sentence spans; split coordination only if both sides fully parse as facts.

    No claim of general atomic extraction. Conditions and causal links stay intact.
    """
    result=[]
    for si,sentence in enumerate(sentences(text),1):
        pieces=re.split(r';\s*|\s+(?:and|while)\s+',sentence)
        parsed=[role_fact(p) for p in pieces]
        if len(pieces)<2 or not all(parsed):pieces=[sentence]
        for part in pieces:result.append({'id':'u'+str(len(result)+1),'sentence_id':si,'text':part.strip()})
    return result


def role_guards(source_units,output_units):
    a=[{'unit_id':u['id'],'fact':role_fact(u['text'])} for u in source_units];b=[{'unit_id':u['id'],'fact':role_fact(u['text'])} for u in output_units]
    a=[r for r in a if r['fact']];b=[r for r in b if r['fact']];conflicts=[]
    # Only compare fully parsed fact groups with the same relation and entity set.
    for relation in ('holding','pay','reimburse'):
        left=[r for r in a if r['fact']['relation']==relation];right=[r for r in b if r['fact']['relation']==relation]
        if not left or not right:continue
        def entities(group):
            return {v for r in group for k,v in r['fact'].items() if k in ('actor','recipient')}
        if len(left)!=len(right) or entities(left)!=entities(right):continue
        canon=lambda group:Counter(json.dumps(r['fact'],sort_keys=True) for r in group)
        if canon(left)!=canon(right):
            conflicts.append({'kind':'quantity_role_conflict' if relation=='holding' else 'actor_role_conflict','source_facts':left,'output_facts':right,
                'scope':'Fully parsed finite grammar; not general entity/coreference resolution'})
    return conflicts


def bundles(parts):
    result=[{'unit_ids':[p['id']],'text':p['text']} for p in parts]
    result += [{'unit_ids':[a['id'],b['id']],'text':a['text'].rstrip('.!?')+'. '+b['text']} for a,b in zip(parts,parts[1:])]
    return result


def verify(source,output,predict):
    a,b=units(source),units(output)
    conflicts=role_guards(a,b)
    surface=claim_checks(source,output)
    conflicts += [e for e in surface['events'] if e['kind']=='scoped_bound_reversal_candidate']
    if not source.strip() or not output.strip() or max(len(a),len(b))>MAX_UNITS:
        return {'status':'abstain','reason':'empty_or_unit_budget','semantic_proof':False,'role_conflicts':conflicts}
    cache={}
    def cached(premise,hypothesis):
        key=(sha(premise),sha(hypothesis))
        if key not in cache:
            if len(cache)>=MAX_PAIRS:return None
            cache[key]=checked_prediction(predict,premise,hypothesis)
        return cache[key]
    global_check=assess(source,output,cached);alignments=[]
    # A best matching local unit or adjacent bundle supports each candidate claim.
    # Both directions matter: source coverage and output support are separate.
    for direction,claims,evidence in (('source_coverage',a,b),('output_support',b,a)):
        candidates=bundles(evidence)
        for claim in claims:
            pairs=[{'evidence_unit_ids':c['unit_ids'],'probabilities':cached(c['text'],claim['text'])} for c in candidates]
            known=[p for p in pairs if p['probabilities'] is not None]
            best=max(known,key=lambda p:p['probabilities']['entailment']) if known else None
            alignments.append({'direction':direction,'claim_unit_id':claim['id'],'claim_text':claim['text'],
                'candidates':pairs,'best_evidence_unit_ids':best['evidence_unit_ids'] if best else None,
                'status':'abstain' if len(known)!=len(pairs) else 'supported_proxy' if best['probabilities']['entailment']>=.8 and best['probabilities']['contradiction']<=.1 else 'flagged_proxy'})
    unknown=global_check['status']=='unknown_over_context' or any(r['status']=='abstain' for r in alignments)
    flagged=global_check['status']=='proxy_flagged' or any(r['status']=='flagged_proxy' for r in alignments)
    status='risk_signal' if conflicts else 'abstain' if unknown else 'risk_signal' if flagged else 'model_consistent'
    return {'status':status,'semantic_proof':False,'global_nli':global_check,'alignments':alignments,
        'role_conflicts':conflicts,'other_surface_warnings':surface['events'],'source_units':a,'output_units':b,
        'prediction_pairs':len(cache),'limitations':'Candidate sentence/clause units, finite role grammar and fallible NLI. Model-consistent is not verified truth; no general claim extraction, coreference or scope guarantee.'}


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
        'limitations':'Authored synthetic controls, not blinded or independently annotated domain accuracy. V1 examples are development-only; only fresh cases are held out.'}
    summary['readiness_gate']=gate(summary['splits']['held_out'],p['readiness_gate']) if execution_kind=='pinned_nli_inference' else {'passed':False,'reason':'Synthetic stub execution cannot validate a verifier'}
    baseline=[{**r,'verdict':{'status':{'proxy_consistent':'model_consistent','proxy_flagged':'risk_signal'}.get(r['verdict'].get('global_nli',{}).get('status'),'abstain')}} for r in records]
    summary['baseline_global_sentence_nli']={s:score([r for r in baseline if r['split']==s]) for s in ('development','held_out')}
    summary['held_out_baseline_passes_overridden']=sum(r['split']=='held_out' and r['verdict'].get('global_nli',{}).get('status')=='proxy_consistent' and r['verdict']['status']=='risk_signal' for r in records)
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
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'docs/claim_verifier_calibration_v2')
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
