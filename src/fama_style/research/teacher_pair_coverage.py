"""Training-only coverage audit of frozen reverse teacher pairs; no generation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics
from .signal import distribution
from fama_style.training.teacher_pair_pilot import sha,prompt

PROTOCOL='teacher_pair_training_coverage_v1'
FEATURES={
 'scope_frequency':'Quantified scope, frequency, universality, negation or exclusions that must retain force.',
 'comparison_ranking':'Ordered comparisons or rankings, not every contrast between two concepts.',
 'event_order_boundary':'Substantive before/after ordering or time boundary; excludes citation years and document roadmap order.',
 'aggregation_horizon':'Temporal aggregation, sampling interval, duration or investment horizon.',
 'theoretical_observed_distinction':'Explicit theoretical/predictive/expected versus observed/realized quantity distinction.',
 'planned_subsequent_investment_contrast':'Explicit distinction between planned investment and subsequent investment; exact error subtype, not all future language.',
 'reported_numeric_values':'Substantive reported values/counts, including number words; excludes citation years, table/section indices and dates alone.',
 'linked_numeric_comparisons':'Multiple reported values with preserved comparative roles, groups, units or periods.',
 'hypothetical_numeric_values':'Numerical hypothetical alternatives or counterfactual values, rather than reported observations.',
 'probability_outcome_tuples':'Explicit numerical probabilities tied to payoffs/outcomes in a choice or indifference example.',
 'condition_assumption':'Stated substantive condition, assumption, threshold or exception.',
 'qualified_modality':'Epistemic/modal force or uncertainty that constrains a conclusion.',
 'stated_reason_mechanism':'A reason, explanation or mechanism actually supplied by the paragraph; does not certify causality.',
 'association_prediction_limit':'Explicit limits on what association, consistency or predictability establishes.'}
FAMILIES={
 'quantifiers_comparisons':['scope_frequency','comparison_ranking'],
 'temporal_distinctions':['event_order_boundary','aggregation_horizon','theoretical_observed_distinction','planned_subsequent_investment_contrast'],
 'numeric_examples':['reported_numeric_values','linked_numeric_comparisons','hypothetical_numeric_values','probability_outcome_tuples'],
 'bounded_explanations':['condition_assumption','qualified_modality','stated_reason_mechanism','association_prediction_limit']}
STATUSES={'aligned_to_archived_text','qualified_alignment','source_wording_uncertainty'}


def validate(records,annotations):
    if len(records)!=46 or len({r['example_id'] for r in records})!=46:
        raise ValueError('exact_46_training_pairs_required')
    by={a['example_id']:a for a in annotations}
    if len(by)!=46 or len(annotations)!=46 or set(by)!={r['example_id'] for r in records}:
        raise ValueError('complete_unique_training_annotations_required')
    for row in records:
        fields={'example_id','paper_id','author_group_id','authors','split','source','target',
                'source_sha256','target_sha256','prompt','pilot_eligible'}
        if set(row)!=fields or row['pilot_eligible'] is not True:raise ValueError('frozen_positive_pair_schema_required')
        if row.get('split')!='training':raise ValueError('training_records_only')
        if (row['source_sha256']!=sha(row['source']) or row['target_sha256']!=sha(row['target'])
                or row['prompt']!=prompt(row['source'])):raise ValueError('frozen_text_or_prompt_changed')
        note=by[row['example_id']]
        if any(note[k]!=row[k] for k in ('source_sha256','target_sha256')):raise ValueError('annotation_binding_changed')
        labels=note['features']
        if (len(labels)!=len(set(labels)) or set(labels)-set(FEATURES)
                or note['alignment_status'] not in STATUSES or not note['coverage_notes']
                or type(note['major_claim_reordering']) is not bool):
            raise ValueError('invalid_or_unexplained_annotation')
        if ('linked_numeric_comparisons' in labels and 'reported_numeric_values' not in labels):
            raise ValueError('linked_values_require_reported_numeric_content')
        if note['alignment_status']!='aligned_to_archived_text' and not note.get('alignment_notes'):
            raise ValueError('alignment_limitation_needs_explanation')
    return by


def analyze(records,annotations):
    by=validate(records,annotations);result=[]
    for row in records:
        note=by[row['example_id']];measurement=metrics(row['source'],row['target'])
        result.append(row|{'annotation':note,'metrics':measurement})
    counts={feature:{'pairs':sum(feature in r['annotation']['features'] for r in result),
        'papers':len({r['paper_id'] for r in result if feature in r['annotation']['features']})} for feature in FEATURES}
    families={name:{'pairs':sum(bool(set(labels)&set(r['annotation']['features'])) for r in result),
        'papers':len({r['paper_id'] for r in result if set(labels)&set(r['annotation']['features'])})} for name,labels in FAMILIES.items()}
    numeric_papers=Counter(r['paper_id'] for r in result if 'reported_numeric_values' in r['annotation']['features'])
    numeric_total=sum(numeric_papers.values())
    summary={'protocol':PROTOCOL,'status':'training_coverage_assessed_no_training',
        'training_pairs':46,'training_papers':len({r['paper_id'] for r in result}),
        'coverage_families':families,'feature_coverage':counts,
        'alignment_status_counts':dict(Counter(r['annotation']['alignment_status'] for r in result)),
        'major_claim_order_changes_in_automated_review':sum(r['annotation']['major_claim_reordering'] for r in result),
        'overlap_distribution':distribution([r['metrics']['word_sequence_overlap'] for r in result],[0,.6,.7,.8,.85,.9,1]),
        'target_source_word_length_ratio_distribution':distribution([r['metrics']['length_ratio'] for r in result]),
        'sentence_count_proxy_changes':sum(r['metrics']['sentence_count_changed'] for r in result),
        'overlap_at_least_0_85':sum(r['metrics']['word_sequence_overlap']>=.85 for r in result),
        'exact_word_token_copies':sum(r['metrics']['word_token_copy'] for r in result),
        'reported_numeric_pairs':numeric_total,'reported_numeric_papers':len(numeric_papers),
        'largest_paper_share_of_reported_numeric_pairs':max(numeric_papers.values())/numeric_total if numeric_total else None,
        'pairs_with_surface_numeric_set_differences':sum(bool(r['metrics']['missing_numeric_values'] or r['metrics']['added_numeric_values']) for r in result),
        'explicit_chosen_rejected_training_pairs':0,'semantic_error_labels_in_training_payload':0,
        'source_or_target_repairs':0,'new_annotations_for_generation':0,
        'development_records_deserialized':0,'reserved_records_deserialized':0,
        'generation_outputs_read':0,'gpu_jobs_submitted':0,'training_authorized':False,
        'limits':['Text-bound automated qualitative annotation, not independent semantic gold or PDF/OCR verification.',
            'Feature presence does not prove that the student learned it or identify the cause of development failures.',
            'Exact error-subtype gaps do not mean all relevant concepts are absent; numeric/style measures are surface proxies.',
            'Positive-only SFT can teach preservation; absent rejected examples are not proof that preference training will help.']}
    return result,summary


def build(curated,annotation_path,output):
    if output.exists():raise ValueError('refuse_overwrite_coverage_audit')
    verify_files(curated)
    records=[json.loads(line) for line in (curated/'training.private.jsonl').read_text().splitlines() if line.strip()]
    annotations=[json.loads(line) for line in annotation_path.read_text().splitlines() if line.strip()]
    result,summary=analyze(records,annotations)
    # Read only matched training audit bodies; no development data or generations.
    ids={r['example_id'] for r in records};audit={}
    for line in (curated/'private_key/training_audit.private.jsonl').read_text().splitlines():
        if not any('"'+rid+'"' in line for rid in ids):continue
        row=json.loads(line)
        if row['example_id'] in ids:audit[row['example_id']]=row
    if set(audit)!=ids:raise ValueError('original_training_audit_coverage')
    for row in result:
        original=audit[row['example_id']]
        if any(row[k]!=original[k] for k in ('source','target','source_sha256','target_sha256')):
            raise ValueError('original_curation_binding_changed')
        row['original_curation']=original
    summary['carried_forward_v1_training_pairs']=sum(r['original_curation']['decision']['decision']=='carry_forward_v1' for r in result)
    summary['pairs_retaining_original_review_flag']=sum(r['original_curation']['original_needs_review'] for r in result)
    summary['coverage_by_curation_origin']={}
    for name,carry in [('original_v1',True),('added_v2',False)]:
        group=[r for r in result if (r['original_curation']['decision']['decision']=='carry_forward_v1')==carry]
        summary['coverage_by_curation_origin'][name]={'pairs':len(group),
            'reported_numeric_pairs':sum('reported_numeric_values' in r['annotation']['features'] for r in group),
            'families':{family:sum(bool(set(labels)&set(r['annotation']['features'])) for r in group) for family,labels in FAMILIES.items()}}
    output.mkdir(parents=True)
    write_rows(output/'per_pair.private.jsonl',result)
    (output/'annotations.private.jsonl').write_bytes(annotation_path.read_bytes())
    provenance=annotation_path.parent/'provenance.private.json'
    if provenance.exists():(output/'provenance.private.json').write_bytes(provenance.read_bytes())
    write_json(output/'summary.json',summary)
    write_json(output/'annotation_specification.json',{'protocol':PROTOCOL,'features':FEATURES,'families':FAMILIES,
        'alignment_statuses':sorted(STATUSES),'annotation_scope':'Both texts assessed using the same labels; labels describe preserved content, not a novel author-style taxonomy.'})
    write_json(output/'receipt.json',{'curation_manifest_sha256':file_sha(curated/'manifest.json'),
        'training_file_sha256':file_sha(curated/'training.private.jsonl'),
        'annotation_sha256':file_sha(annotation_path),
        'metrics_source_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py')),
        'read_scope':'training.private.jsonl and matching training-audit rows; other release files hash-verified only'})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('curated','annotations','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.curated,args.annotations,args.output)))


if __name__=='__main__':main()
