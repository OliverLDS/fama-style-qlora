"""Exploratory CPU selection of previously proposed, locally verified merges.

Whole-paper outer/inner folds; predictions use the existing twelve source-only
features. Ties and claim-shift judgments never become classifier training labels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from . import boundary_classifier as logistic
from . import source_only_grouping as source_features
from .author_comparison_corpus import words
from .author_control_replication import verify_files
from .human_prose_review import normalized
from .relation_pilot_review import file_manifest, file_sha, read_rows
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'verified_merge_selector_v1'
CONFIG = {'protocol': PROTOCOL, 'features': list(source_features.FEATURES),
    'primary': 'faithful_and_decisive_readability_only',
    'excluded_from_fit': ['faithful_comparable', 'claim_shift', 'unresolved'],
    'outer_folds': 'leave_one_whole_paper_out', 'inner_folds': 'leave_one_other_paper_out',
    'penalty': 1.0, 'class_weight': 'none', 'probability_thresholds': [0.35, 0.5, 0.65, 0.8, 1.0],
    'length_thresholds': [0, 20, 30, 40, 50, 60, 80, 160],
    'length_rule': 'merge_if_source_pair_regex_word_count_at_most_threshold',
    'selection': 'macro_paper_accuracy_then_fewer_merges_then_more_conservative_threshold',
    'methods': ['keep', 'merge_all', 'length_only', 'source_logistic'],
    'fitted_scaler': False, 'candidate_or_judge_text_in_features': False,
    'configuration_locked_before_predictions_not_prospective_preregistration': True}


def cohort(row):
    if row['fidelity_verdict'] in ('local_shift', 'material_shift'): return 'claim_shift', None
    if row['fidelity_verdict'] != 'faithful' or row['readability_in_context']=='unresolved': return 'unresolved', None
    if row['readability_in_context']=='comparable': return 'faithful_comparable', None
    return 'primary', int(row['readability_in_context']=='merged_better')


def predictor_view(source):
    """Whitelist actual source text and segmentation; no claims or judgments."""
    sentences = [s['exact_text'] for s in source['source_sentences']]
    if not sentences or normalized(' '.join(sentences)) != normalized(source['source_paragraph']):
        raise ValueError('source_segmentation_changed')
    return {'sentences': sentences}


def collect(verified, sources):
    mapping = {r['item_id']:r for r in json.loads((verified/'PRIVATE_MAPPING.json').read_text())}
    originals = {r['paragraph_id']:r for r in read_rows(sources/'annotations.private.jsonl')}
    results = []
    for row in read_rows(verified/'annotations.private.jsonl'):
        m = mapping[row['item_id']]; source = originals[m['paragraph_id']]
        view = predictor_view(source); index = int(m['boundary_id'][1:])-1
        pair = view['sentences'][index:index+2]
        if len(pair)!=2 or pair != [row['left_sentence'],row['right_sentence']] or row['source_paragraph']!=source['source_paragraph']:
            raise ValueError('source_pair_changed')
        group,label = cohort(row)
        results.append({'item_id': row['item_id'], 'paper_id': m['paper_id'], 'paragraph_id': m['paragraph_id'],
            'boundary_id': m['boundary_id'], 'features': source_features.boundary_features(view)[index],
            'pair_words': sum(len(words(s)) for s in pair), 'cohort': group, 'label': label,
            'fidelity': row['fidelity_verdict'], 'readability': row['readability_in_context']})
    if len({r['item_id'] for r in results})!=len(results): raise ValueError('duplicate_item')
    return results


def score(rows):
    primary = [r for r in rows if r['cohort']=='primary']
    tp=sum(r['label']==r['prediction']==1 for r in primary)
    fp=sum(r['label']==0 and r['prediction']==1 for r in primary)
    fn=sum(r['label']==1 and r['prediction']==0 for r in primary)
    tn=sum(r['label']==r['prediction']==0 for r in primary)
    papers=sorted({r['paper_id'] for r in primary})
    return {'primary_items': len(primary), 'selected_primary_merges': tp+fp,
        'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'accuracy':(tp+tn)/len(primary) if primary else None,
        'macro_paper_accuracy': statistics.mean(sum(r['label']==r['prediction'] for r in primary if r['paper_id']==p)/sum(r['paper_id']==p for r in primary) for p in papers) if papers else None,
        'merge_precision':tp/(tp+fp) if tp+fp else None, 'merge_recall':tp/(tp+fn) if tp+fn else None,
        'merge_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'side_cohorts': {c:{'items':sum(r['cohort']==c for r in rows),
            'selected_merges':sum(r['cohort']==c and r['prediction']==1 for r in rows),
            'selected_readability':dict(Counter(r['readability'] for r in rows if r['cohort']==c and r['prediction']==1))}
            for c in ('faithful_comparable','claim_shift','unresolved')}}


def fit_primary(rows):
    if not rows or any(r['cohort']!='primary' or r['label'] not in (0,1) for r in rows):
        raise ValueError('only_decisive_faithful_training_rows')
    return logistic.fit(rows, CONFIG['penalty'], CONFIG['class_weight'])


def choose(candidates, conservative):
    return max(candidates,key=lambda c:(c['metrics']['macro_paper_accuracy'],
        -c['metrics']['selected_primary_merges'], conservative(c['threshold'])))


def tune(train, forbidden_paper):
    """Only outer-training primary labels select thresholds; no side outcomes."""
    if any(r['paper_id']==forbidden_paper for r in train): raise ValueError('outer_paper_in_tuning')
    if any(r['cohort']!='primary' for r in train): raise ValueError('side_cohort_in_tuning')
    papers=sorted({r['paper_id'] for r in train})
    if len(papers)!=3: raise ValueError('three_inner_papers_required')
    held_probabilities={};trace=[]
    for held in papers:
        fitting=[r for r in train if r['paper_id']!=held]; query=[r for r in train if r['paper_id']==held]
        model=fit_primary(fitting)
        trace.append({'inner_paper':held,'training_papers':sorted({r['paper_id'] for r in fitting}),
            'training_items':len(fitting),'iterations':model['iterations'],'converged':model['converged']})
        held_probabilities.update({r['item_id']:logistic.probability(model,r['features']) for r in query})
    candidates=[]
    for t in CONFIG['probability_thresholds']:
        predictions=[{**r,'prediction':int(held_probabilities[r['item_id']]>=t)} for r in train]
        candidates.append({'threshold':t,'metrics':score(predictions)})
    lengths=[]
    for t in CONFIG['length_thresholds']:
        # No estimated coefficients: inner held-out length predictions equal the fixed rule.
        predictions=[{**r,'prediction':int(r['pair_words']<=t)} for r in train]
        lengths.append({'threshold':t,'metrics':score(predictions)})
    return {'source_logistic':choose(candidates,lambda t:t),
            'length_only':choose(lengths,lambda t:-t),
            'logistic_candidates':candidates,'length_candidates':lengths,'inner_fit_trace':trace}


def evaluate(rows):
    papers=sorted({r['paper_id'] for r in rows})
    if len(papers)!=4: raise ValueError('four_outer_papers_required')
    predictions=[];folds=[]
    for held in papers:
        train=[r for r in rows if r['paper_id']!=held and r['cohort']=='primary']
        query=[r for r in rows if r['paper_id']==held]
        selection=tune(train,held);model=fit_primary(train)
        folds.append({'outer_paper':held,'training_papers':sorted({r['paper_id'] for r in train}),
            'training_primary_items':len(train),'selection':selection,'outer_model':model})
        for r in query:
            probability=logistic.probability(model,r['features'])
            choices={'keep':0,'merge_all':1,'length_only':int(r['pair_words']<=selection['length_only']['threshold']),
                     'source_logistic':int(probability>=selection['source_logistic']['threshold'])}
            for method,prediction in choices.items():
                predictions.append({**{k:r[k] for k in ('item_id','paper_id','paragraph_id','boundary_id','cohort','label','fidelity','readability')},
                    'method':method,'prediction':prediction,
                    'probability':probability if method=='source_logistic' else None})
    return predictions,folds


def protocol(verified,sources):
    return {'config':CONFIG,'code_sha256':file_sha(Path(__file__)),
        'solver_sha256':file_sha(Path(logistic.__file__)),
        'feature_code_sha256':file_sha(Path(source_features.__file__)),
        'verification_manifest_sha256':file_sha(verified/'manifest.json'),
        'source_manifest_sha256':file_sha(sources/'manifest.json')}


def run(verified,sources,protocol_path,output):
    verify_files(verified);verify_files(sources)
    if json.loads(protocol_path.read_text())!=protocol(verified,sources): raise ValueError('locked_protocol_changed')
    for audit in (verified,sources):
        if json.loads((audit/'summary.json').read_text())['invalid_records']: raise ValueError('invalid_input_records')
    rows=collect(verified,sources)
    if len(rows)!=52 or Counter(r['cohort'] for r in rows)!={'primary':35,'faithful_comparable':9,'claim_shift':8}:
        raise ValueError('frozen_sample_changed')
    predictions,folds=evaluate(rows)
    methods={m:score([r for r in predictions if r['method']==m]) for m in CONFIG['methods']}
    summary={'protocol':PROTOCOL,'status':'exploratory_cpu_evaluation_complete','config':CONFIG,
        'items':len(rows),'papers':4,'cohorts':dict(Counter(r['cohort'] for r in rows)),
        'primary_label_counts':dict(Counter('merge' if r['label'] else 'keep' for r in rows if r['cohort']=='primary')),
        'methods':methods,'per_paper':{p:{m:score([r for r in predictions if r['paper_id']==p and r['method']==m]) for m in CONFIG['methods']} for p in sorted({r['paper_id'] for r in rows})},
        'selected_thresholds':{f['outer_paper']:{m:f['selection'][m]['threshold'] for m in ('length_only','source_logistic')} for f in folds},
        'outer_fits':len(folds),'inner_fits':sum(len(f['selection']['inner_fit_trace']) for f in folds),
        'all_fits_converged':all(f['outer_model']['converged'] and all(t['converged'] for t in f['selection']['inner_fit_trace']) for f in folds),
        'paper_overlap_between_fit_and_query':False,'query_labels_used_for_tuning':False,
        'candidate_or_judge_text_in_features':False,'source_segmentation':'frozen_Stage2_source_only_sentences',
        'labels_are_independent_ground_truth':False,'selected_proposal_set_only':True,
        'local_predictions_composed_into_paragraph_edits':False,'classical_cpu_classifier_fitted':True,
        'automatic_training_acceptance':False,'new_model_calls':0,'gpu_jobs_started':0,
        'neural_or_adapter_training_called':False,'reserved_content_used':False}
    fresh(output);write_json(output/'summary.json',summary)
    write_lines(output/'features_and_outcomes.private.jsonl',rows)
    write_lines(output/'predictions.private.jsonl',predictions)
    write_json(output/'folds.private.json',folds)
    (output/'protocol.json').write_bytes(protocol_path.read_bytes())
    file_manifest(output,**protocol(verified,sources))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('verified','sources','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=run(a.verified,a.sources,a.protocol,a.output);print(json.dumps(s['methods'],indent=2))


if __name__=='__main__':main()
