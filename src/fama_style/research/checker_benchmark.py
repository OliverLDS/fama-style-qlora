"""Frozen CPU benchmark for claim-risk detection, false alarms and abstention."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .claim_audit import claim_checks

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='claim_checker_benchmark_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rate(a,b):return a/b if b else None


def classify(audit):
    if audit['events']:return 'risk_signal'
    return 'surface_accept' if audit['status']=='surface_replay_only' else 'abstain'


def summarize(records):
    faithful=[r for r in records if r['expected']=='faithful'];changed=[r for r in records if r['expected']=='changed']
    hard=[r for r in faithful if r['variant']=='paraphrase']
    return {'rows':len(records),'faithful_rows':len(faithful),'changed_rows':len(changed),
        'prediction_counts':dict(Counter(r['prediction'] for r in records)),
        'error_detection_rate':rate(sum(r['prediction']=='risk_signal' for r in changed),len(changed)),
        'faithful_false_alarm_rate':rate(sum(r['prediction']=='risk_signal' for r in faithful),len(faithful)),
        'faithful_surface_accept_rate':rate(sum(r['prediction']=='surface_accept' for r in faithful),len(faithful)),
        'nontrivial_faithful_surface_accept_rate':rate(sum(r['prediction']=='surface_accept' for r in hard),len(hard)),
        'changed_surface_accept_rate':rate(sum(r['prediction']=='surface_accept' for r in changed),len(changed)),
        'abstention_rate':rate(sum(r['prediction']=='abstain' for r in records),len(records)),
        'changed_abstention_rate':rate(sum(r['prediction']=='abstain' for r in changed),len(changed)),
        'faithful_abstention_rate':rate(sum(r['prediction']=='abstain' for r in faithful),len(faithful)),
        'nontrivial_faithful_false_alarm_rate':rate(sum(r['prediction']=='risk_signal' for r in hard),len(hard))}


def passes_gate(metrics,gate):
    checks={
        'error_detection':metrics['error_detection_rate'] is not None and metrics['error_detection_rate']>=gate['minimum_error_detection_rate'],
        'faithful_false_alarms':metrics['faithful_false_alarm_rate'] is not None and metrics['faithful_false_alarm_rate']<=gate['maximum_faithful_false_alarm_rate'],
        'faithful_acceptance':metrics['faithful_surface_accept_rate'] is not None and metrics['faithful_surface_accept_rate']>=gate['minimum_faithful_surface_accept_rate'],
        'nontrivial_acceptance':metrics['nontrivial_faithful_surface_accept_rate'] is not None and metrics['nontrivial_faithful_surface_accept_rate']>=gate['minimum_nontrivial_faithful_surface_accept_rate']}
    return {'passed':all(checks.values()),'checks':checks}


def run(protocol_dir,output):
    if output.exists():raise ValueError('Use a fresh output directory')
    protocol_path=protocol_dir/'frozen_protocol.json';p=json.loads(protocol_path.read_text())
    if p['protocol']!=PROTOCOL:raise ValueError('Wrong benchmark protocol')
    paths={ROOT/name:h for name,h in p['checker_files'].items()}
    paths.update({protocol_dir/name:h for name,h in p['fixture_files'].items()})
    if not all(digest(path)==h for path,h in paths.items()):raise ValueError('Frozen checker or fixtures changed; define a new benchmark version')
    groups={};cases=[]
    for split in ('development','held_out'):
        rows=json.loads((protocol_dir/(split+'.json')).read_text())
        if len(rows)!=36 or any(r['split']!=split for r in rows):raise ValueError('Wrong benchmark split')
        groups[split]={r['scenario_group'] for r in rows};cases+=rows
    if groups['development']&groups['held_out'] or len({r['id'] for r in cases})!=72:raise ValueError('Overlapping scenario groups or IDs')
    if {r['source'] for r in cases if r['split']=='development'}&{r['source'] for r in cases if r['split']=='held_out'}:raise ValueError('Leaked source scenario')
    records=[]
    # Development evaluated first; no learned thresholds or rule mutations occur.
    for c in cases:
        audit=claim_checks(c['source'],c['output'])
        records.append({**c,'prediction':classify(audit),'audit':audit})
    summary={'protocol':PROTOCOL,'execution_kind':'synthetic_checker_evaluation','model_calls':0,'gpu_jobs_submitted':0,
        'reserved_paper_test_rows_used':0,'rules_tuned_on_benchmark':False,'checker_files':p['checker_files'],
        'splits':{split:summarize([r for r in records if r['split']==split]) for split in groups},
        'held_out_by_category':{cat:summarize([r for r in records if r['split']=='held_out' and r['category']==cat]) for cat in sorted({r['category'] for r in records})},
        'provenance':p['provenance'],'limitations':'Small authored synthetic controls; not blinded, independent annotator labels or economics-domain accuracy. Surface acceptance is not semantic proof. Held-out scenarios have different wording but shared categories.'}
    summary['readiness_gate']=passes_gate(summary['splits']['held_out'],p['readiness_gate'])
    for name,choice in (('reject_everything','risk_signal'),('abstain_everything','abstain')):
        m=summarize([{**r,'prediction':choice} for r in records if r['split']=='held_out'])
        summary.setdefault('degenerate_controls',{})[name]={'metrics':m,'gate':passes_gate(m,p['readiness_gate'])}
    if not all(digest(path)==h for path,h in paths.items()):raise ValueError('Inputs changed during benchmark')
    output.mkdir(parents=True)
    for name,data in (('results.json',records),('summary.json',summary)):(output/name).write_text(json.dumps(data,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'input_hashes':{str(path):h for path,h in paths.items()},
        'protocol_sha256':digest(protocol_path),'runner_sha256':digest(Path(__file__)),
        'files':{name:digest(output/name) for name in ('results.json','summary.json')}},indent=2)+'\n')
    print(json.dumps({'splits':summary['splits'],'readiness_gate':summary['readiness_gate']},indent=2));return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol-dir',type=Path,default=ROOT/'docs/claim_checker_benchmark_v1')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv);run(a.protocol_dir,a.output_dir)


if __name__=='__main__':main()
