"""Offline style/drift tradeoffs from saved candidates; no binary NLI veto."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .claims import sha
from .signal import distance,overlap

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='style_drift_tradeoff_v1'


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def nli_pressure(verdict):
    """Continuous warning pressure, not calibrated semantic severity."""
    if not verdict or 'global_nli' not in verdict:return {'complete':False,'omission':None,'addition':None,'contradiction':None}
    g=verdict['global_nli'];omission=[];addition=[];contra=[];complete=True
    for row in g['checks']:
        p=row['probabilities']
        if p is None:complete=False;continue
        d=row['direction'];contra.append(p['contradiction'])
        if d=='whole_reverse' or d.startswith('source_unit_'):omission.append(1-p['entailment'])
        if d=='whole_forward' or d.startswith('output_unit_'):addition.append(1-p['entailment'])
    for a in verdict.get('alignments',[]):
        known=[r['probabilities'] for r in a['candidates'] if r['probabilities'] is not None]
        if len(known)!=len(a['candidates']):complete=False
        if not known:continue
        best=max(known,key=lambda p:p['entailment']);contra.append(best['contradiction'])
        (omission if a['direction']=='source_coverage' else addition).append(1-best['entailment'])
    return {'complete':complete and bool(omission and addition),'omission':max(omission) if omission else None,
        'addition':max(addition) if addition else None,'contradiction':max(contra) if contra else None}


def equity_role_frame(text):
    # Same explicit incentive-alignment frame, not a whole-paragraph word count.
    pattern=r'\b(?:incentives|interests)\b[^.;!?]{0,100}?\b(?:with|of)\b[^.;!?]{0,55}?\bother\s+(shareholders|stockholders|security[ -]holders)\b'
    return [{'group':'equity_holders' if m.group(1).lower() in ('shareholders','stockholders') else 'security_holders',
             'evidence':m.group()} for m in re.finditer(pattern,text,re.I)]


def high_cost_evidence(source,output,verdict):
    result=[]
    if verdict:
        result += [{'kind':x['kind'],'provenance':'frozen_verifier_finite_guard','evidence':x} for x in verdict.get('role_conflicts',[])]
        result += [{'kind':x['kind'],'provenance':'frozen_scoped_relation_guard','evidence':x} for x in verdict.get('scoped_relation_guards',{}).get('conflicts',[])]
    a,b=equity_role_frame(source),equity_role_frame(output)
    if any(x['group']=='equity_holders' for x in a) and any(x['group']=='security_holders' for x in b):
        result.append({'kind':'incentive_actor_category_broadened','provenance':'exploratory_literal_role_frame',
            'source_evidence':a,'output_evidence':b,'verified_semantic_error':False})
    return result


def score_candidate(source,output,profile,verdict,policy):
    before,after=distance(source,profile),distance(output,profile)
    benefit=1/(1+after)-1/(1+before);same=' '.join(source.split())==' '.join(output.split())
    pressure=nli_pressure(verdict);typed=high_cost_evidence(source,output,verdict)
    if same:cost=0.;pressure={'complete':True,'omission':0.,'addition':0.,'contradiction':0.};typed=[]
    else:
        components=[pressure[k] for k in ('omission','addition','contradiction')]
        cost=sum(policy['proxy_pressure_weights'][k]*pressure[k] for k in ('omission','addition','contradiction')) if all(x is not None for x in components) else None
        if typed:cost=max(cost or 0.,policy['typed_high_cost_risk_floor'])
    warning=('high_cost_risk_candidate' if typed else 'severity_unlocalized' if verdict and verdict.get('status')!='model_consistent' else 'no_high_cost_signal_detected')
    return {'style_distance_gain':before-after,'style_alignment_benefit_proxy':benefit,'source_style_distance':before,'output_style_distance':after,
        'nli_pressure':pressure,'semantic_cost_proxy':cost,'severity_evidence_category':warning,'high_cost_evidence':typed,
        'true_drift_severity':'unknown','semantic_proof':False,'old_verifier_status':verdict.get('status') if verdict else 'not_available',
        'source_similarity':overlap(source,output),'near_copy':overlap(source,output)>=.9,'exact_copy':same,
        'proxy_utilities':{name:benefit-weight*cost if cost is not None else None for name,weight in policy['cost_weights'].items()},
        'ranking_uncertainty':'incomplete_semantic_evidence' if not pressure['complete'] else 'uncalibrated_proxy'}


def frontier(candidates,benefit_key='style_alignment_benefit_proxy',cost_key='semantic_cost_proxy'):
    """Within-case only. Unknown-cost candidates remain separate, not rejected."""
    result=[]
    for c in candidates:
        if c[cost_key] is None:continue
        dominated=any(d[cost_key] is not None and d[benefit_key]>=c[benefit_key] and d[cost_key]<=c[cost_key]
            and (d[benefit_key]>c[benefit_key] or d[cost_key]<c[cost_key]) for d in candidates)
        if not dominated:result.append(c['candidate_id'])
    return result


def collect():
    base=ROOT/'research_runs';sources={};observations=[];invalid=[];inputs={};profile=None
    runs=[('soft_guidance_validation_v2','soft_guidance_validation_v2','condition'),
          ('ledger_grouping_ablation_v1','ledger_grouping_ablation_v1','condition'),
          ('sentence_preserving_edit_v1','sentence_preserving_edit_v1','variant')]
    for folder,runname,field in runs:
        root=base/folder;p=root/'kaggle_version_1'/runname;review_path=root/'version_1_review/verification.json';review=json.loads(review_path.read_text())
        if not review['checks'] or not all(review['checks'].values()):raise ValueError('Previous verification failed')
        for name,h in review['artifact_hashes'].items():
            if file_sha(p/name)!=h:raise ValueError('Previously reviewed artifact changed')
        pack_name={'soft_guidance_validation_v2':'soft_guidance_pack.json','ledger_grouping_ablation_v1':'ledger_ablation_pack.json','sentence_preserving_edit_v1':'sentence_edit_pack.json'}[folder]
        packpath=root/'input_pack'/pack_name;pack=json.loads(packpath.read_text());release=json.loads((root/'release_verification.json').read_text())
        if file_sha(packpath)!=release['pack_sha256']:raise ValueError('Frozen pack changed')
        if profile is None:profile=pack['style_profile']
        if profile!=pack['style_profile']:raise ValueError('Different style profiles cannot be pooled')
        for r in pack['rows']:
            if r['split']!='validation':raise ValueError('Only development validation allowed')
            s={'source':r['source'],'reference':r['reference']}
            if r['id'] in sources and sources[r['id']]!=s:raise ValueError('Source/reference mismatch')
            sources[r['id']]=s
        outputs={(r['id'],r['condition']):r for r in read_lines(p/'generations.private.jsonl')}
        for m in read_lines(p/'metrics.private.jsonl'):
            text=m.get('paragraph') if field=='variant' else outputs[(m['id'],m['condition'])]['output']
            provenance={'experiment':folder,'version':1,'condition':m[field],'source_id':m['id']}
            if text is None:invalid.append({**provenance,'reason':'invalid_script'});continue
            observations.append({'source_id':m['id'],'output':text,'verdict':m['verifier'],'provenance':provenance})
        for path in (packpath,review_path,p/'metrics.private.jsonl',p/'generations.private.jsonl'):
            inputs[str(path.relative_to(ROOT))]=file_sha(path)
    return sources,profile,observations,invalid,inputs


def assess_tradeoff(style, drift, policy):
    """Explicit ordinal judgments; never infer severity from NLI confidence."""
    if style not in policy['style_rubric'] or drift not in policy['drift_rubric']:
        raise ValueError('Unknown rubric label')
    benefit=policy['qualitative_style_values'].get(style)
    cost=policy['qualitative_drift_values'].get(drift)
    threshold=policy['qualitative_style_values'][policy['minimum_style_for_substantial_candidate']]
    return {'assessed_style_value':benefit,'assessed_drift_cost':cost,
        'substantial_style_candidate':None if benefit is None else benefit>=threshold,
        'assessed_utilities':{name:benefit-weight*cost if benefit is not None and cost is not None else None
            for name,weight in policy['cost_weights'].items()}}


def apply_assessments(candidates, packet, policy):
    """Bind exploratory AI judgments to exact texts; leave unaudited rows unknown."""
    if packet.get('protocol')!=PROTOCOL or not packet.get('provenance'):
        raise ValueError('Assessment provenance required')
    indexed={c['candidate_id']:c for c in candidates};seen=set();result=[]
    for a in packet['assessments']:
        cid=a['candidate_id']
        if cid in seen or cid not in indexed:raise ValueError('Duplicate or unknown candidate')
        seen.add(cid);c=indexed[cid]
        for field in ('source','reference','output'):
            if a[field+'_sha256']!=sha(c[field]):raise ValueError('Assessment text changed')
        if not a.get('style_evidence') or not a.get('drift_evidence'):
            raise ValueError('Separate style and semantic evidence required')
        result.append({**a,**assess_tradeoff(a['style_grade'],a['drift_grade'],policy)})
    return result


def rescore(output,assessments=None):
    if output.exists():raise ValueError('Use a fresh output directory')
    policy_path=ROOT/'docs/style_drift_tradeoff_v1/policy.json';policy=json.loads(policy_path.read_text())
    sources,profile,obs,invalid,inputs=collect();unique={}
    for row in obs:
        source=sources[row['source_id']]['source'];key=(sha(source),sha(row['output']))
        if key not in unique:
            unique[key]={'candidate_id':sha('|'.join(key))[:20],'source_id':row['source_id'],'source':source,
                'reference':sources[row['source_id']]['reference'],'output':row['output'],'observations':[],'verdicts':[]}
        unique[key]['observations'].append(row['provenance']);unique[key]['verdicts'].append(row['verdict'])
    candidates=[]
    for c in unique.values():
        scored=[score_candidate(c['source'],c['output'],profile,v,policy) for v in c['verdicts']]
        # Repeated controls are not independent samples; use maximum recorded pressure.
        chosen=max(scored,key=lambda s:-1 if s['semantic_cost_proxy'] is None else s['semantic_cost_proxy'])
        candidates.append({**c,**chosen,'old_verifier_statuses':sorted({v['status'] for v in c['verdicts']}),
            'verifier_replicate_disagreement':len({v['status'] for v in c['verdicts']})>1})
    controls=[]
    for id,s in sources.items():
        controls.append({'candidate_id':'source-'+sha(s['source'])[:16],'source_id':id,'source':s['source'],'reference':s['reference'],'output':s['source'],
            'observations':[],'is_source_control':True,**score_candidate(s['source'],s['source'],profile,None,policy)})
    fronts={id:frontier([c for c in candidates+controls if c['source_id']==id]) for id in sources}
    positive=[c for c in candidates if c['style_distance_gain']>0]
    summary={'protocol':PROTOCOL,'execution_kind':'offline_saved_output_rescoring','model_calls':0,'gpu_jobs_submitted':0,'training_called':False,
        'reserved_test_rows_read':0,'observations':len(obs)+len(invalid),'available_views':len(obs),'invalid_views':len(invalid),'unique_candidates':len(candidates),'source_controls':len(controls),
        'source_cases':len(sources),'positive_style_proxy_candidates':len(positive),
        'old_risk_or_abstain_candidates_retained':sum(c['old_verifier_status']!='model_consistent' for c in candidates),
        'high_cost_risk_candidates':sum(bool(c['high_cost_evidence']) for c in candidates),
        'within_case_frontier_candidates':sum(c['candidate_id'] in fronts[c['source_id']] for c in candidates),
        'limitations':'Exploratory post-hoc reranking of reused development outputs; no semantic-severity calibration or author-style accuracy. NLI pressure is not amount of true claim drift. Duplicate/derived views are not independent samples.'}
    judged=[]
    if assessments is not None:
        packet=json.loads(assessments.read_text());judged=apply_assessments(candidates,packet,policy)
        inputs[str(assessments.relative_to(ROOT)) if assessments.is_relative_to(ROOT) else assessments.name]=file_sha(assessments)
        summary['qualitative_assessment']={'assessed_candidates':len(judged),'unassessed_candidates':len(candidates)-len(judged),
            'style_grades':dict(Counter(a['style_grade'] for a in judged)),
            'drift_grades':dict(Counter(a['drift_grade'] for a in judged)),
            'substantial_style_candidates':sum(a['substantial_style_candidate'] is True for a in judged),
            'provenance':packet['provenance'],
            'limitations':'Unblinded exploratory assistant judgments of a proxy-selected subset; not calibrated, independent or population-wide accuracy.'}
    output.mkdir(parents=True)
    if assessments is not None:
        (output/'assessments.private.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in judged))
    for name,values in (('candidates.private.jsonl',candidates),('source_controls.private.jsonl',controls),('invalid_views.private.jsonl',invalid)):
        (output/name).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in values))
    (output/'frontiers.private.json').write_text(json.dumps(fronts,indent=2)+'\n');(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'policy_sha256':file_sha(policy_path),'input_hashes':inputs,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}},indent=2)+'\n')
    print(json.dumps(summary,indent=2));return candidates,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--assessments',type=Path);a=p.parse_args(argv);rescore(a.output_dir,a.assessments)


if __name__=='__main__':main()
