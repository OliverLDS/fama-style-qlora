"""Stdlib ridge logistic boundary classifier with nested whole-paper holdouts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from . import source_only_grouping as baseline
from .author_control_replication import verify_files
from .author_comparison_corpus import file_sha
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='boundary_classifier_v1'
CONFIG={'protocol':PROTOCOL,'features':list(baseline.FEATURES),'feature_changes_from_retriever':False,
    'penalties':[0.01,0.1,1.0],'class_weights':['none','balanced'],'thresholds':[0.35,0.5,0.65],
    'objective':'weighted_mean_log_loss_plus_half_lambda_squared_slope_norm',
    'intercept_penalized':False,'scaling':'inherited_fixed_caps_no_fitted_scaler',
    'outer_folds':'leave_one_paper_out','inner_folds':'leave_one_other_paper_out',
    'selection':'pooled_inner_merge_f1_then_precision_then_macro_paper_exact_then_stronger_penalty_then_higher_threshold_then_unweighted',
    'solver':'Newton_with_backtracking','max_iterations':80,'gradient_tolerance':1e-7,
    'heldout_split':'training_discovery_only','candidates':18,'tuning_outside_inner_folds':False}


def sigmoid(z):
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def sample_weights(labels,kind):
    if kind not in ('none','balanced'):raise ValueError('class_weight')
    counts=Counter(labels);n=len(labels)
    return [1.0 if kind=='none' else n/(len(counts)*counts[y]) for y in labels]


def objective(weights,data,penalty):
    """Mean weighted log loss, analytic gradient and Hessian; intercept at zero."""
    d=len(weights);gradient=[0.0]*d;hessian=[[0.0]*d for _ in range(d)]
    total=sum(weight for _,_,weight in data);loss=0.0
    for active,y,weight in data:
        z=sum(weights[i]*x for i,x in active);p=sigmoid(z);factor=weight/total
        loss+=factor*(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
        for j,(i,x) in enumerate(active):
            gradient[i]+=factor*(p-y)*x
            for k,v in active[j:]:hessian[i][k]+=factor*p*(1-p)*x*v
    for i in range(d):
        for j in range(i):hessian[i][j]=hessian[j][i]
        if i:
            loss+=penalty*weights[i]**2/2;gradient[i]+=penalty*weights[i];hessian[i][i]+=penalty
    return loss,gradient,hessian


def solve(matrix,vector):
    n=len(vector);a=[list(row)+[value] for row,value in zip(matrix,vector)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(a[j][i]))
        if abs(a[pivot][i])<1e-14:raise ValueError('singular_hessian')
        a[i],a[pivot]=a[pivot],a[i];divisor=a[i][i]
        for k in range(i,n+1):a[i][k]/=divisor
        for j in range(n):
            if j==i:continue
            factor=a[j][i]
            for k in range(i,n+1):a[j][k]-=factor*a[i][k]
    return [row[-1] for row in a]


def fit(rows,penalty,class_weight):
    if not rows or penalty<=0:raise ValueError('training_rows_or_penalty')
    d=len(rows[0]['features']);labels=[r['label'] for r in rows]
    if any(y not in (0,1) for y in labels) or any(len(r['features'])!=d for r in rows):raise ValueError('training_shape')
    if any(not math.isfinite(v) for r in rows for v in r['features']):raise ValueError('nonfinite_features')
    positive=sum(labels);n=len(labels)
    weights=[math.log((positive+0.5)/(n-positive+0.5))]+[0.0]*d
    if len(set(labels))==1:
        return {'weights':weights,'penalty':penalty,'class_weight':class_weight,'iterations':0,
                'converged':True,'single_class_smoothed_constant':True}
    sw=sample_weights(labels,class_weight)
    data=[([(0,1.0)]+[(i+1,x) for i,x in enumerate(r['features']) if x],r['label'],w) for r,w in zip(rows,sw)]
    for iteration in range(CONFIG['max_iterations']):
        loss,gradient,hessian=objective(weights,data,penalty)
        if max(abs(g) for g in gradient)<=CONFIG['gradient_tolerance']:
            return {'weights':weights,'penalty':penalty,'class_weight':class_weight,'iterations':iteration,
                    'converged':True,'single_class_smoothed_constant':False}
        delta=solve(hessian,gradient);slope=sum(g*v for g,v in zip(gradient,delta));step=1.0
        while step>=2**-30:
            proposed=[w-step*v for w,v in zip(weights,delta)]
            if objective(proposed,data,penalty)[0]<=loss-1e-4*step*slope:break
            step/=2
        else:raise ValueError('line_search_failed')
        weights=proposed
    raise ValueError('solver_not_converged')


def probability(model,features):
    if len(features)+1!=len(model['weights']):raise ValueError('prediction_shape')
    return sigmoid(model['weights'][0]+sum(w*x for w,x in zip(model['weights'][1:],features)))


def boundary_rows(examples):
    result=[]
    for e in examples:
        view=e['source']
        for i,(f,y) in enumerate(zip(baseline.boundary_features(view),e['labels'])):
            result.append({'paper_id':view['paper_id'],'paragraph_id':view['paragraph_id'],
                           'boundary':i,'features':f,'label':y})
    return result


def score_probabilities(examples,probabilities,threshold):
    rows=[]
    for e in examples:
        view=e['source'];pid=view['paragraph_id']
        values=probabilities[pid]
        if len(values)!=len(e['labels']):raise ValueError('probability_coverage')
        rows.append({'paragraph_id':pid,'paper_id':view['paper_id'],'gold_labels':e['labels'],
            'predicted_labels':[int(p>=threshold) for p in values]})
    return baseline.score(rows)


def selection_key(candidate):
    s=candidate['inner_metrics']
    return (s['merge_f1'] or 0,s['merge_precision'] or 0,s['macro_paper_exact_rate'] or 0,
            candidate['penalty'],candidate['threshold'],candidate['class_weight']=='none')


def tune(examples,forbidden_paper):
    if any(e['source']['paper_id']==forbidden_paper for e in examples):raise ValueError('outer_paper_in_tuning')
    papers=sorted({e['source']['paper_id'] for e in examples})
    if len(papers)<3:raise ValueError('inner_papers')
    candidates=[];traces=[]
    for penalty in CONFIG['penalties']:
        for weighting in CONFIG['class_weights']:
            predictions={}
            for heldout in papers:
                train=[e for e in examples if e['source']['paper_id']!=heldout]
                query=[e for e in examples if e['source']['paper_id']==heldout]
                model=fit(boundary_rows(train),penalty,weighting)
                traces.append({'inner_paper':heldout,'training_papers':sorted({e['source']['paper_id'] for e in train}),
                    'penalty':penalty,'class_weight':weighting,'iterations':model['iterations'],'converged':model['converged']})
                for e in query:
                    predictions[e['source']['paragraph_id']]=[probability(model,f) for f in baseline.boundary_features(e['source'])]
            for threshold in CONFIG['thresholds']:
                candidates.append({'penalty':penalty,'class_weight':weighting,'threshold':threshold,
                                   'inner_metrics':score_probabilities(examples,predictions,threshold)})
    return max(candidates,key=selection_key),candidates,traces


def evaluate(examples):
    papers=sorted({e['source']['paper_id'] for e in examples});predictions=[];folds=[]
    for heldout in papers:
        train=[e for e in examples if e['source']['paper_id']!=heldout]
        query=[e for e in examples if e['source']['paper_id']==heldout]
        chosen,candidates,traces=tune(train,heldout)
        model=fit(boundary_rows(train),chosen['penalty'],chosen['class_weight'])
        folds.append({'outer_paper':heldout,'training_papers':sorted({e['source']['paper_id'] for e in train}),
            'selected':chosen,'candidates':candidates,'inner_fit_trace':traces,'outer_model':model})
        for e in query:
            view=e['source'];values=[probability(model,f) for f in baseline.boundary_features(view)]
            labels=[int(p>=chosen['threshold']) for p in values]
            predictions.append({'paragraph_id':view['paragraph_id'],'paper_id':heldout,'method':'nested_logistic',
                'probabilities':values,'predicted_labels':labels,'gold_labels':e['labels'],
                'predicted_source_sentence_groups':baseline.plan_from_labels(view,labels)})
    return predictions,folds


def run(audit_directory,frozen_baseline,protocol_path,output):
    verify_files(audit_directory);verify_files(frozen_baseline)
    protocol=json.loads(protocol_path.read_text())
    if protocol['config']!=CONFIG or protocol['code_sha256']!=file_sha(Path(__file__)):
        raise ValueError('frozen_classifier_protocol_changed')
    if protocol['baseline_code_sha256']!=file_sha(Path(baseline.__file__)) or protocol['audit_manifest_sha256']!=file_sha(audit_directory/'manifest.json'):
        raise ValueError('upstream_changed')
    records=[json.loads(l) for l in (audit_directory/'audits.private.jsonl').read_text().splitlines()]
    audited,examples=baseline.collect(records)
    if len(audited)!=60 or len(examples)!=57 or len({e['source']['paper_id'] for e in examples})!=12:raise ValueError('frozen_selection')
    previous_audit=[json.loads(l) for l in (frozen_baseline/'alignment_audit.private.jsonl').read_text().splitlines()]
    if audited!=previous_audit:raise ValueError('eligibility_changed')
    predictions,folds=evaluate(examples)
    old=[json.loads(l) for l in (frozen_baseline/'predictions.private.jsonl').read_text().splitlines()]
    if {r['paragraph_id'] for r in predictions}!={r['paragraph_id'] for r in old}:raise ValueError('baseline_comparison_coverage')
    models={method:baseline.score([r for r in old if r['method']==method]) for method in baseline.METHODS}
    models['nested_logistic']=baseline.score(predictions)
    summary={'protocol':PROTOCOL,'status':'nested_cpu_evaluation_complete_no_generation','config':CONFIG,
        'eligible_rows':57,'eligible_papers':12,'boundaries':models['nested_logistic']['boundaries'],
        'methods':models,'per_paper':{p:{**{m:baseline.score([r for r in old if r['paper_id']==p and r['method']==m]) for m in baseline.METHODS},
            'nested_logistic':baseline.score([r for r in predictions if r['paper_id']==p])} for p in sorted({r['paper_id'] for r in predictions})},
        'selected_configuration_counts':dict(Counter(str((f['selected']['penalty'],f['selected']['class_weight'],f['selected']['threshold'])) for f in folds)),
        'outer_fits':len(folds),'inner_fits':sum(len(f['inner_fit_trace']) for f in folds),
        'all_fits_converged':all(f['outer_model']['converged'] and all(t['converged'] for t in f['inner_fit_trace']) for f in folds),
        'outer_paper_used_in_inner_selection':False,'query_reference_in_predictor':False,
        'source_segmentation_origin':'joint paired GPT-6 annotations; not independent source extraction',
        'eligibility_reference_informed':True,'generated_prose_quality_evaluated':False,'distinctive_author_style_established':False,
        'development_rows_scored':0,'reserved_rows_used':0,'original_fama_splits_used':0,
        'external_model_calls':0,'gpu_jobs_started':0,'classical_cpu_classifier_fitted':True,'neural_or_adapter_training_called':False}
    fresh(output);write_json(output/'summary.json',summary);write_json(output/'folds.private.json',folds)
    write_lines(output/'predictions.private.jsonl',predictions)
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'baseline_code_sha256':file_sha(Path(baseline.__file__)),'audit_manifest_sha256':file_sha(audit_directory/'manifest.json'),
        'frozen_baseline_manifest_sha256':file_sha(frozen_baseline/'manifest.json'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','frozen-baseline','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=run(a.audit,a.frozen_baseline,a.protocol,a.output);print(json.dumps(s['methods'],indent=2))


if __name__=='__main__':main()
