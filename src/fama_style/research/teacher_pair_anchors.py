"""Freeze training-only faithful variants and single-edit semantic counterexamples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics,numbers
from .teacher_pair_coverage import FAMILIES as COVERAGE_FAMILIES
from fama_style.training.teacher_pair_pilot import sha,prompt

PROTOCOL='teacher_pair_semantic_anchors_v1'
FAMILIES={
 'scope':'quantifiers_comparisons','time':'temporal_distinctions',
 'numbers':'numeric_examples','reasoning':'bounded_explanations'}
ERRORS={
 'scope':{'scope_reversal','unsupported_modal_strengthening','unsupported_scope_strengthening'},
 'time':{'ordering_reversal','boundary_reversal','future_past_substitution'},
 'numbers':{'value_role_swap','magnitude_change'},
 'reasoning':{'unsupported_reason_substitution','unsupported_causal_link'}}
CONSTRAINTS={'quantities','time','conditions','negation','modality','causality'}


def construct(draft,original):
    if original.get('split')!='training':raise ValueError('training_only_anchors')
    if (draft['source_sha256']!=sha(original['source']) or draft['target_sha256']!=sha(original['target'])
            or original['source_sha256']!=draft['source_sha256'] or original['target_sha256']!=draft['target_sha256']):
        raise ValueError('frozen_anchor_binding_changed')
    variant=draft['faithful_variant']
    if not isinstance(variant,str) or not variant.strip() or ' '.join(variant.split()) in {
            ' '.join(original['source'].split()),' '.join(original['target'].split())}:
        raise ValueError('nontrivial_faithful_variant_required')
    # Numbers are a necessary screen, never a proof of equivalence or correct roles.
    if set(numbers(variant))!=set(numbers(original['source'])):
        raise ValueError('variant_numeric_set_changed')
    for name in draft['protected_names']:
        if any(name.casefold() not in text.casefold() for text in (original['source'],original['target'],variant)):
            raise ValueError('protected_name_missing')
    claims=draft['claims'];ids={c['claim_id'] for c in claims}
    if len(claims)<2 or len(ids)!=len(claims):raise ValueError('unique_claim_inventory_required')
    for claim in claims:
        if (not claim['statement'] or not claim['actors'] or not claim['relation']
                or set(claim['constraints'])!=CONSTRAINTS
                or any(not isinstance(v,list) or any(not isinstance(s,str) or not s for s in v) for v in claim['constraints'].values())):
            raise ValueError('structured_claim_constraints_required')
        for key,text in [('source_evidence',original['source']),('target_evidence',original['target']),('variant_evidence',variant)]:
            evidence=claim[key]
            if not evidence or evidence.casefold() not in text.casefold():raise ValueError('claim_evidence_span_missing')
    edit=draft['negative_edit'];before=edit['before'];after=edit['after']
    if not before or not after or before==after or original['target'].count(before)!=1:
        raise ValueError('single_unambiguous_target_edit_required')
    if (draft['family'] not in FAMILIES or edit['error_type'] not in ERRORS[draft['family']]
            or not edit['explanation'] or not edit['affected_claim_ids']
            or not set(edit['affected_claim_ids'])<=ids):raise ValueError('negative_claim_binding_required')
    rejected=original['target'].replace(before,after,1)
    return {'example_id':original['example_id'],'paper_id':original['paper_id'],
        'author_group_id':original['author_group_id'],'authors':original['authors'],'split':'training',
        'family':draft['family'],'source':original['source'],'source_sha256':draft['source_sha256'],
        'faithful_variant':variant,'variant_sha256':sha(variant),'prompt':prompt(variant),
        'chosen':original['target'],'target_sha256':draft['target_sha256'],
        'rejected':rejected,'rejected_sha256':sha(rejected),
        'claims':claims,'negative_edit':edit,'protected_names':draft['protected_names'],
        'training_ready':False,'semantic_gold_verified':False,'synthetic_variant':True,'synthetic_rejection':True}


def validate_review(row,review):
    if any(review[k]!=row[k] for k in ('example_id','variant_sha256','target_sha256','rejected_sha256')):
        raise ValueError('review_text_binding_changed')
    if (review.get('claim_ledger_sha256')!=sha(json.dumps(row['claims'],sort_keys=True,ensure_ascii=False))
            or review.get('negative_edit_sha256')!=sha(json.dumps(row['negative_edit'],sort_keys=True,ensure_ascii=False))):
        raise ValueError('review_claim_ledger_or_edit_contract_changed')
    expected={c['claim_id'] for c in row['claims']};checks=review['claim_checks']
    if (len(checks)!=len(expected) or {c['claim_id'] for c in checks}!=expected
            or any(c['preserved'] is not True or not c['notes'] for c in checks)
            or review['added_claim_identified'] is not False
            or review['negative_is_unfaithful'] is not True or not review['review_notes']):
        raise ValueError('complete_provisional_semantic_review_required')


def build(coverage,drafts_path,reviews_path,output):
    if output.exists():raise ValueError('refuse_overwrite_anchor_package')
    verify_files(coverage)
    # Coverage per-pair file contains training only; never open a development input.
    originals={}
    for line in (coverage/'per_pair.private.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('split')!='training':raise ValueError('coverage_must_be_training_only')
        if row['example_id'] in originals:raise ValueError('duplicate_training_record')
        originals[row['example_id']]=row
    drafts=[json.loads(l) for l in drafts_path.read_text().splitlines() if l.strip()]
    reviews=[json.loads(l) for l in reviews_path.read_text().splitlines() if l.strip()]
    ids={d['example_id'] for d in drafts};review_by={r['example_id']:r for r in reviews}
    if (len(drafts)!=12 or len(ids)!=12 or not ids<=set(originals)
            or len(reviews)!=12 or len(review_by)!=12 or set(review_by)!=ids
            or Counter(d['family'] for d in drafts)!=Counter({f:3 for f in FAMILIES})):
        raise ValueError('twelve_unique_training_anchors_three_per_family_required')
    anchors=[]
    for draft in drafts:
        original=originals[draft['example_id']]
        if original['annotation']['alignment_status']!='aligned_to_archived_text':raise ValueError('exclude_recorded_alignment_limits')
        if not set(original['annotation']['features'])&set(COVERAGE_FAMILIES[FAMILIES[draft['family']]]):
            raise ValueError('anchor_family_not_supported_by_coverage')
        row=construct(draft,original);validate_review(row,review_by[row['example_id']])
        row['review']=review_by[row['example_id']];row['original_curation']=original['original_curation']
        anchors.append(row)
    if len({r['faithful_variant'] for r in anchors})!=12 or len({r['chosen'] for r in anchors})!=12:
        raise ValueError('duplicate_variant_or_target')
    measurements=[metrics(r['faithful_variant'],r['chosen']) for r in anchors]
    rejected_metrics=[metrics(r['chosen'],r['rejected']) for r in anchors]
    summary={'protocol':PROTOCOL,'status':'provisional_data_candidates_not_training_release',
        'anchors':12,'papers':len({r['paper_id'] for r in anchors}),
        'families':dict(Counter(r['family'] for r in anchors)),'faithful_variants':12,'counterexamples':12,
        'claims':sum(len(r['claims']) for r in anchors),'original_human_targets_unchanged':True,
        'anchors_retaining_original_review_flag':sum(r['original_curation']['original_needs_review'] for r in anchors),
        'single_contiguous_negative_edits':12,
        'negative_edits_preserving_numeric_value_multisets':sum(numbers(r['chosen'])==numbers(r['rejected']) for r in anchors),
        'median_variant_target_word_sequence_overlap':statistics.median(m['word_sequence_overlap'] for m in measurements),
        'median_original_source_target_overlap':statistics.median(metrics(r['source'],r['chosen'])['word_sequence_overlap'] for r in anchors),
        'minimum_rejected_chosen_word_sequence_overlap':min(m['word_sequence_overlap'] for m in rejected_metrics),
        'negative_word_length_ratio_range':[min(m['length_ratio'] for m in rejected_metrics),max(m['length_ratio'] for m in rejected_metrics)],
        'major_original_selection_changes':0,'development_records_read':0,'reserved_records_read':0,
        'external_annotation_calls':0,'gpu_jobs_submitted':0,'training_authorized':False,
        'full_scale_training_ready':False,'tokenizer_preflight':'not_run',
        'limits':['Variants, ledgers and negative judgments are constructed and second-pass reviewed by the same Codex reviewer; not independent semantic gold.',
            'Exact edit reconstruction and lexical/numeric screens verify integrity, not entailment or style quality.',
            'One role-swap operation may affect two linked claims. Unsupported strengthened claims or reasons need not be universally factually false.',
            'Synthetic errors may contain shortcuts and are not a natural-error benchmark; existing eight-case development exposure remains.',
            'This package is not wired into training or a Kaggle notebook and does not choose SFT versus a preference objective.']}
    output.mkdir(parents=True)
    write_rows(output/'anchors.private.jsonl',anchors)
    write_rows(output/'sft_candidates.private.jsonl',[{'example_id':r['example_id'],'split':'training',
        'source':r['faithful_variant'],'prompt':r['prompt'],'completion':r['chosen'],
        'variant_sha256':r['variant_sha256'],'target_sha256':r['target_sha256'],'training_ready':False} for r in anchors])
    write_rows(output/'preference_candidates.private.jsonl',[{k:r[k] for k in ('example_id','split','prompt','chosen','rejected','variant_sha256','target_sha256','rejected_sha256','training_ready')} for r in anchors])
    (output/'construction.private.jsonl').write_bytes(drafts_path.read_bytes())
    (output/'review.private.jsonl').write_bytes(reviews_path.read_bytes())
    for name in ('provenance.private.json','construction_revision.private.json','construction_draft_v0.private.jsonl'):
        path=drafts_path.parent/name
        if path.exists():(output/name).write_bytes(path.read_bytes())
    write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'coverage_manifest_sha256':file_sha(coverage/'manifest.json'),
        'construction_sha256':file_sha(drafts_path),'review_sha256':file_sha(reviews_path),
        'source_target_policy':'Original targets exact; only synthetic variants and one-edit rejected targets are new.'})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('coverage','drafts','reviews','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.coverage,args.drafts,args.reviews,args.output)))


if __name__=='__main__':main()
