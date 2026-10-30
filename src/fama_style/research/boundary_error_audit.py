"""Versioned, evidence-bound discovery audit; never relabel or rescore the classifier."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .author_control_replication import verify_files
from .author_comparison_corpus import file_sha,words
from .human_prose_review import occurrence_span
from .source_only_grouping import plan_from_labels
from fama_style.rhetoric.experiment import fresh,write_json,write_lines

PROTOCOL='boundary_error_audit_v1'
CATEGORIES={'plausible_alternative':{'extra'},'keep_preferred_risk':{'extra'},
    'supported_merge_missed':{'missed'},'optional_reference_merge':{'missed'},
    'composite_claim_granularity':{'missed','extra'}}


def make_cards(predictions,annotations):
    cards=[]
    for prediction in predictions:
        row=annotations[prediction['paragraph_id']]
        if row['split']!='training':raise ValueError('training_only_audit')
        a=row['annotation'];units=a['source_schema']['sentences']
        gold=prediction['gold_labels'];pred=prediction['predicted_labels']
        if len(gold)!=len(pred) or len(gold)!=len(units)-1:raise ValueError('boundary_shape')
        if any(type(v) is not int or v not in (0,1) for v in gold+pred):raise ValueError('binary_labels')
        groups=plan_from_labels({'sentences':units},pred)
        if groups!=prediction['predicted_source_sentence_groups']:raise ValueError('prediction_group_mismatch')
        for i,(g,p) in enumerate(zip(gold,pred)):
            if g==p:continue
            left,right=units[i],units[i+1];claim_ids=set(left['claim_ids'])|set(right['claim_ids'])
            target=[s for s in a['target_schema']['sentences'] if set(s['claim_ids'])&claim_ids]
            group=next(group for group in groups if i+1 in group)
            identity=prediction['paragraph_id']+':'+str(i)
            cards.append({'case_id':hashlib.sha256((PROTOCOL+identity).encode()).hexdigest()[:16],
                'ordinal':len(cards)+1,'paragraph_id':prediction['paragraph_id'],'paper_id':prediction['paper_id'],
                'boundary':i,'kind':'extra' if p else 'missed','gold_label':g,'predicted_label':p,
                'source':a['plain_english'],'reference':a['original_paragraph'],
                'left':left['exact_text'],'right':right['exact_text'],
                'left_claim_ids':left['claim_ids'],'right_claim_ids':right['claim_ids'],
                'shared_claim_ids':sorted(set(left['claim_ids'])&set(right['claim_ids'])),
                'target_sentences':[s['exact_text'] for s in target],
                'predicted_group_containing_left':group,
                'predicted_group_source_words':sum(len(words(units[j-1]['exact_text'])) for j in group),
                'source_needs_external_context':a['source_quality']['needs_external_context'],
                'annotation_review_notes':a['review_notes']})
    if len({c['case_id'] for c in cards})!=len(cards):raise ValueError('duplicate_case')
    return cards


def validate_assessment(card,assessment):
    if assessment['category'] not in CATEGORIES or card['kind'] not in CATEGORIES[assessment['category']]:
        raise ValueError('category_kind')
    if not assessment['rationale'].strip() or not assessment['relation'].strip():raise ValueError('missing_explanation')
    if type(assessment['confirmed_label_error']) is not bool:raise ValueError('label_error_boolean')
    if not assessment['source_evidence'] or not assessment['reference_evidence']:raise ValueError('missing_evidence')
    for key,text in (('source_evidence',card['source']),('reference_evidence',card['reference'])):
        if any(occurrence_span(text,e) is None for e in assessment[key]):raise ValueError('evidence_not_exact')
    if not any(e['quote']==card['left'] for e in assessment['source_evidence']) or not any(e['quote']==card['right'] for e in assessment['source_evidence']):
        raise ValueError('both_source_sides_required')
    if assessment['category']=='composite_claim_granularity' and not card['shared_claim_ids']:
        raise ValueError('no_composite_claim_basis')
    if assessment['confirmed_label_error'] and not assessment.get('label_error_explanation','').strip():
        raise ValueError('label_error_requires_explanation')


def run(classifier,audit,assessments,output):
    classifier_manifest=verify_files(classifier);verify_files(audit)
    if classifier_manifest['audit_manifest_sha256']!=file_sha(audit/'manifest.json'):raise ValueError('classifier_audit_mismatch')
    predictions=[json.loads(l) for l in (classifier/'predictions.private.jsonl').read_text().splitlines()]
    annotations={r['paragraph_id']:r for r in map(json.loads,(audit/'audits.private.jsonl').read_text().splitlines())}
    cards=make_cards(predictions,annotations)
    if Counter(c['kind'] for c in cards)!={'extra':20,'missed':22}:raise ValueError('frozen_error_set')
    spec=json.loads(assessments.read_text())
    if spec['protocol']!=PROTOCOL or spec['classifier_manifest_sha256']!=file_sha(classifier/'manifest.json'):
        raise ValueError('assessment_provenance')
    by_id={a['case_id']:a for a in spec['records']}
    if len(by_id)!=len(spec['records']) or set(by_id)!={c['case_id'] for c in cards}:raise ValueError('assessment_coverage')
    for card in cards:validate_assessment(card,by_id[card['case_id']])
    records=[{'card':c,'assessment':by_id[c['case_id']]} for c in cards]
    summary={'protocol':PROTOCOL,'status':'discovery_audit_complete_no_rescoring',
        'cases':len(cards),'paragraphs':len({c['paragraph_id'] for c in cards}),
        'papers':len({c['paper_id'] for c in cards}),
        'by_kind':{kind:{'cases':sum(c['kind']==kind for c in cards),
            'categories':dict(Counter(by_id[c['case_id']]['category'] for c in cards if c['kind']==kind))} for kind in ('extra','missed')},
        'by_paper':{paper:dict(Counter(by_id[c['case_id']]['category'] for c in cards if c['paper_id']==paper)) for paper in sorted({c['paper_id'] for c in cards})},
        'keep_preferred_unique_predicted_groups':len({(c['paragraph_id'],tuple(c['predicted_group_containing_left'])) for c in cards if by_id[c['case_id']]['category']=='keep_preferred_risk'}),
        'confirmed_label_errors':sum(a['confirmed_label_error'] for a in by_id.values()),
        'assessment_method':spec['assessment_method'],'assessment_blinded':False,'independent_verifier':False,
        'generated_prose_reviewed':False,'judgments_certified':False,
        'labels_changed':False,'classifier_predictions_changed':False,'classifier_rescored':False,
        'discovery_only_not_new_heldout_score':True,'development_rows_used':0,'reserved_rows_used':0,
        'new_model_fits':0,'external_model_calls':0,'gpu_jobs_started':0,'llm_training_called':False}
    fresh(output);write_lines(output/'cases.private.jsonl',records);write_json(output/'summary.json',summary)
    (output/'assessments.private.json').write_bytes(assessments.read_bytes())
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'classifier_manifest_sha256':file_sha(classifier/'manifest.json'),'audit_manifest_sha256':file_sha(audit/'manifest.json'),
        'assessment_sha256':file_sha(assessments),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('classifier','audit','assessments','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.classifier,a.audit,a.assessments,a.output),indent=2))


if __name__=='__main__':main()
