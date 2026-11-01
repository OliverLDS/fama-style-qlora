"""Curate training-only preferences from real frozen adapter errors; no model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from fama_style.evaluation import semantic_anchor_4b as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics

PROTOCOL='actual_error_preference_candidates_v1'
STAGE='expanded_epoch_2'


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def normalized(text):return ' '.join(text.split())


def validate_anchor(anchor,training,review):
    rid=anchor['example_id']
    if rid not in training or anchor['split']!='training' or training[rid]['split']!='training':
        raise ValueError('training_membership_required')
    parent=training[rid]
    if (anchor['paper_id']!=parent['paper_id'] or anchor['author_group_id']!=parent['author_group_id']
            or anchor['authors']!=parent['authors'] or anchor['source']!=parent['source'] or anchor['chosen']!=parent['target']
            or anchor['source_sha256']!=runtime.sha(parent['source']) or anchor['target_sha256']!=runtime.sha(parent['target'])
            or parent['source_sha256']!=anchor['source_sha256'] or parent['target_sha256']!=anchor['target_sha256']
            or review['target_sha256']!=anchor['target_sha256'] or not review['alignment'] or not review['reference_style']):
        raise ValueError('fixed_source_or_human_target_changed')
    ledger=runtime.sha(json.dumps(anchor['claims'],sort_keys=True,ensure_ascii=False))
    if review['claim_ledger_sha256']!=ledger or anchor['review']['claim_ledger_sha256']!=ledger:
        raise ValueError('claim_ledger_changed')
    claims=anchor['claims'];checks=anchor['review']['claim_checks']
    if (not claims or len({c['claim_id'] for c in claims})!=len(claims) or len(checks)!=len(claims)
            or {c['claim_id'] for c in checks}!={c['claim_id'] for c in claims}
            or not all(c['preserved'] is True for c in checks) or anchor['review']['added_claim_identified'] is not False):
        raise ValueError('complete_local_claim_alignment_required')
    for c in claims:
        for field,text in [('source_evidence',anchor['source']),('variant_evidence',anchor['faithful_variant']),('target_evidence',anchor['chosen'])]:
            if not c[field].strip() or normalized(c[field]) not in normalized(text):raise ValueError('claim_evidence_span_missing')


def decision(anchor,generation,review):
    kind=generation['input_kind'];source=anchor['source'] if kind=='original' else anchor['faithful_variant']
    if kind not in ('original','variant'):raise ValueError('unexpected_input_kind')
    judged=review[kind]
    if (generation['example_id']!=anchor['example_id'] or generation['paper_id']!=anchor['paper_id']
            or generation['stage']!=STAGE or generation['job_id']!=anchor['example_id']+'__'+kind
            or generation['source']!=source or generation['prompt']!=runtime.prompt(source)
            or generation['target_sha256']!=anchor['target_sha256']
            or generation['assessment']['text_sha256']!=runtime.sha(generation['output'])
            or judged['source_sha256']!=runtime.sha(source) or judged['output_sha256']!=runtime.sha(generation['output'])
            or generation['assessment']['material_claim_issue_identified']!=judged['material_error']
            or generation['assessment']['severity_boundary_uncertain']!=judged['uncertain']
            or type(judged['material_error']) is not bool or type(judged['uncertain']) is not bool
            or not judged['reason'] or not judged['style_notes']):raise ValueError('review_or_actual_generation_binding_failed')
    if (generation['empty']!=(not bool(generation['output'].strip()))
            or generation['copied_source']!=(normalized(generation['output'])==normalized(source))):
        raise ValueError('generation_flags_changed')
    if generation['empty'] or generation['output_limited']:outcome='hold_generation_failure'
    elif judged['uncertain']:outcome='hold_uncertain_boundary'
    elif not judged['material_error']:outcome='exclude_no_clear_material_error'
    elif normalized(anchor['chosen'])==normalized(generation['output']):raise ValueError('chosen_and_actual_rejected_must_differ')
    elif judged['style_comparison']!='reference_preferred_provisionally':outcome='hold_style_preference_unestablished'
    else:outcome='include_fidelity_preference_candidate'
    return {'example_id':anchor['example_id'],'job_id':generation['job_id'],'input_kind':kind,'paper_id':anchor['paper_id'],
        'family':anchor['family'],'decision':outcome,'target_sha256':anchor['target_sha256'],
        'source_sha256':runtime.sha(source),'actual_output_sha256':runtime.sha(generation['output']),
        'reason':judged['reason'],'style_comparison':judged['style_comparison'],'style_notes':judged['style_notes'],
        'semantic_style_length_confounded':True,'historical_material_issue':generation['assessment']['material_claim_issue_identified'],
        'historical_boundary_uncertain':generation['assessment']['severity_boundary_uncertain']}


def make_pair(anchor,generation,judgment):
    chosen=anchor['chosen'];rejected=generation['output'];kind=generation['input_kind'];source=generation['source']
    if judgment['decision']!='include_fidelity_preference_candidate':raise ValueError('excluded_decisions_cannot_become_pairs')
    if rejected==anchor['rejected']:raise ValueError('synthetic_negative_cannot_replace_actual_generation')
    return {'pair_id':'AP_'+runtime.sha(PROTOCOL+'|'+generation['job_id'])[:16],'split':'training',
        'example_id':anchor['example_id'],'anchor_group_id':anchor['example_id'],'paper_group_id':anchor['paper_id'],
        'paper_id':anchor['paper_id'],'author_group_id':anchor['author_group_id'],'authors':anchor['authors'],
        'family':anchor['family'],'input_kind':kind,'source':source,'prompt':generation['prompt'],
        'source_sha256':runtime.sha(source),'prompt_sha256':runtime.sha(generation['prompt']),
        'chosen':chosen,'chosen_sha256':runtime.sha(chosen),'rejected':rejected,'rejected_sha256':runtime.sha(rejected),
        'chosen_origin':'unchanged_training_human_reference','rejected_origin':'archived_first_greedy_expanded_epoch_2_generation',
        'synthetic_rejection':False,'generation_job_id':generation['job_id'],'generation_stage':STAGE,
        'preference_basis':'unambiguous_material_error_in_actual_rejection_with_locally_supported_human_target',
        'style_preference':'reference_preferred_provisionally','style_isolated':False,
        'inherited_needs_review':anchor['original_curation']['original_needs_review'],
        'inherited_needs_external_context':anchor['original_curation']['original_needs_external_context'],
        'inherited_review_notes':anchor['original_curation']['upstream_review_notes'],
        'semantic_gold_verified':False,'tokenizer_preflight':'pending_for_actual_rejection','training_ready':False}


def build(training,anchors,packet,download,prior_review,bound_review,output):
    if output.exists():raise ValueError('refuse_overwrite_preference_packet')
    for folder in (anchors,packet,prior_review):verify_files(folder)
    # Read only the parent training payload. Do not parse development/test bodies.
    training_manifest=json.loads((training/'manifest.json').read_text())
    if training_manifest['files']['training.private.jsonl']!=file_sha(training/'training.private.jsonl'):
        raise ValueError('parent_training_changed')
    parents=rows(training/'training.private.jsonl');by_parent={r['example_id']:r for r in parents}
    if len(parents)!=46 or len(by_parent)!=46 or any(r['split']!='training' for r in parents):raise ValueError('original_46_training_rows_required')
    if file_sha(anchors/'anchors.private.jsonl')!=file_sha(packet/'private_key/anchors.private.jsonl'):
        raise ValueError('anchor_inference_lineage_changed')
    cases=rows(anchors/'anchors.private.jsonl');by_anchor={r['example_id']:r for r in cases}
    if len(cases)!=12 or len(by_anchor)!=12 or Counter(a['family'] for a in cases)!=Counter({f:3 for f in ('scope','time','numbers','reasoning')}):
        raise ValueError('original_twelve_balanced_anchors_required')
    jobs,config=runtime.load_inputs(packet/'upload/semantic_anchor_jobs.jsonl',packet/'upload/semantic_anchor_config.json')
    old_receipt=json.loads((prior_review/'receipt.json').read_text())
    if old_receipt['packet_manifest_sha256']!=file_sha(packet/'manifest.json'):raise ValueError('wrong_verified_inference_packet')
    for name,expected in old_receipt['download_sha256'].items():
        if file_sha(download/name)!=expected:raise ValueError('archived_run_changed')
    report=json.loads((download/'semantic_anchor_4b_v1/evaluation.json').read_text())
    if (report['config']!=config or report.get('complete') is not True or report.get('training_called') is not False
            or report['runtime']['model_id']!=runtime.MODEL_ID or report['runtime']['model_revision']!=runtime.REVISION):
        raise ValueError('pinned_completed_inference_required')
    snapshots=[s for s in report['snapshots'] if s['name']==STAGE]
    if len(snapshots)!=1 or not snapshots[0]['complete']:raise ValueError('current_adapter_snapshot_required')
    raw={r['job_id']:r for r in snapshots[0]['generations']}
    previous=[r for r in rows(prior_review/'per_output.private.jsonl') if r['stage']==STAGE]
    by_previous={r['job_id']:r for r in previous};expected={r['job_id'] for r in jobs}
    if len(raw)!=24 or len(previous)!=24 or len(by_previous)!=24 or set(raw)!=expected or set(by_previous)!=expected:
        raise ValueError('all_twenty_four_current_adapter_generations_required')
    review=json.loads(bound_review.read_text());by_review={r['example_id']:r for r in review['cases']}
    if (review['protocol']!='actual_error_preference_review_v1' or len(review['cases'])!=12 or set(by_review)!=set(by_anchor)
            or review['independent_review'] is not False or review['semantic_gold_verified'] is not False):
        raise ValueError('complete_provisional_source_first_review_required')
    decisions=[];pairs=[];alignment=[]
    for anchor in cases:
        judged=by_review[anchor['example_id']];validate_anchor(anchor,by_parent,judged)
        alignment.append({'example_id':anchor['example_id'],'target_sha256':anchor['target_sha256'],
            'claim_ledger_sha256':judged['claim_ledger_sha256'],'claim_ids':[c['claim_id'] for c in anchor['claims']],
            'all_evidence_spans_present':True,'local_alignment':'supported_for_original_and_variant_inputs',
            'alignment_notes':judged['alignment'],'reference_style':judged['reference_style'],
            'independent_verification':False,'inherited_curation':anchor['original_curation']})
        for kind in ('original','variant'):
            g=by_previous[anchor['example_id']+'__'+kind]
            if any(g[k]!=raw[g['job_id']][k] for k in raw[g['job_id']]):raise ValueError('reviewed_generation_differs_from_raw')
            d=decision(anchor,g,judged);decisions.append(d)
            if d['decision']=='include_fidelity_preference_candidate':pairs.append(make_pair(anchor,g,d))
    anchor_counts=Counter(p['anchor_group_id'] for p in pairs);paper_counts=Counter(p['paper_group_id'] for p in pairs)
    for p in pairs:
        p.update(anchor_balanced_weight=1/anchor_counts[p['anchor_group_id']],paper_balanced_weight=1/paper_counts[p['paper_group_id']])
    audit=json.loads((download/'semantic_anchor_4b_v1/label_audit.json').read_text())
    chosen_audits={r['job_id']:r for r in audit if r['candidate']=='chosen'}
    diagnostics=[]
    for p in pairs:
        chosen_rejected=metrics(p['chosen'],p['rejected']);source_chosen=metrics(p['source'],p['chosen']);source_rejected=metrics(p['source'],p['rejected'])
        a=chosen_audits[p['generation_job_id']];g=by_previous[p['generation_job_id']]
        if a['decoded_target_matches'] is not True or a['prompt_tokens']+g['output_tokens']>runtime.MAX_LENGTH:
            raise ValueError('archived_generation_budget_evidence_failed')
        diagnostics.append({'pair_id':p['pair_id'],'example_id':p['example_id'],'input_kind':p['input_kind'],
            'chosen_rejected':chosen_rejected,'source_chosen':source_chosen,'source_rejected':source_rejected,
            'archived_prompt_tokens':a['prompt_tokens'],'archived_chosen_full_tokens':a['full_tokens'],
            'archived_generation_prompt_plus_output_tokens':a['prompt_tokens']+g['output_tokens'],
            'new_rejected_completion_mask_verified':False,
            'chosen_to_rejected_word_ratio':chosen_rejected['reference_words']/max(1,chosen_rejected['output_words'])})
    sft=[{k:p[k] for k in ('pair_id','example_id','anchor_group_id','paper_group_id','split','prompt','prompt_sha256','anchor_balanced_weight','paper_balanced_weight')}|
        {'completion':p['chosen'],'completion_sha256':p['chosen_sha256'],'training_ready':False} for p in pairs]
    summary={'protocol':PROTOCOL,'status':'curated_candidates_not_training_release','candidates_assessed':24,'included_pairs':len(pairs),
        'decisions':dict(Counter(d['decision'] for d in decisions)),'unique_included_anchors':len(anchor_counts),'included_papers':len(paper_counts),
        'families':dict(Counter(p['family'] for p in pairs)),'input_kinds':dict(Counter(p['input_kind'] for p in pairs)),
        'reviewed_claims':sum(len(a['claims']) for a in cases),'chosen_targets_modified':0,'synthetic_rejections_used':0,
        'exact_source_copy_rejections':sum(normalized(p['source'])==normalized(p['rejected']) for p in pairs),
        'exact_source_copy_chosen':sum(normalized(p['source'])==normalized(p['chosen']) for p in pairs),
        'included_with_inherited_review_flags':sum(p['inherited_needs_review'] for p in pairs),
        'paired_sft_control_rows':len(sft),'chosen_longer_than_rejected':sum(d['chosen_to_rejected_word_ratio']>1 for d in diagnostics),
        'median_chosen_to_rejected_word_ratio':statistics.median(d['chosen_to_rejected_word_ratio'] for d in diagnostics) if diagnostics else None,
        'chosen_to_rejected_word_ratio_range':[min(d['chosen_to_rejected_word_ratio'] for d in diagnostics),max(d['chosen_to_rejected_word_ratio'] for d in diagnostics)] if diagnostics else [],
        'median_source_chosen_overlap':statistics.median(d['source_chosen']['word_sequence_overlap'] for d in diagnostics) if diagnostics else None,
        'median_source_rejected_overlap':statistics.median(d['source_rejected']['word_sequence_overlap'] for d in diagnostics) if diagnostics else None,
        'median_chosen_rejected_overlap':statistics.median(d['chosen_rejected']['word_sequence_overlap'] for d in diagnostics) if diagnostics else None,
        'chosen_more_source_overlap_than_rejected':sum(d['source_chosen']['word_sequence_overlap']>d['source_rejected']['word_sequence_overlap'] for d in diagnostics),
        'max_archived_chosen_full_tokens':max(d['archived_chosen_full_tokens'] for d in diagnostics) if diagnostics else None,
        'max_archived_prompt_plus_generated_tokens':max(d['archived_generation_prompt_plus_output_tokens'] for d in diagnostics) if diagnostics else None,
        'development_records_parsed':0,'reserved_records_parsed':0,'independent_semantic_gold':False,'distinctive_fama_style_labels':False,
        'detector_labels_used_for_selection':False,'external_calls':0,'model_calls':0,'training_authorized':False,'gpu_jobs_submitted':0,
        'tokenizer_preflight':'not_run_for_actual_rejected_outputs','training_ready':False,
        'limits':['Candidate rows remain correlated by original paragraph and paper; report group counts, not independent sample counts.',
            'Chosen references were already SFT targets; preference versus additional-SFT effects require a matched chosen-only control.',
            'Meaning, reference wording, generic style and response length differ jointly; these are not clean style-only or Fama-style labels.',
            'Same reviewer knows historical generations and created variants; local alignment is provisional, not independent factual/transcription verification.',
            'Actual rejection log-probabilities and token boundaries are not available from the earlier synthetic-negative score audit.']}
    protocol={'objective':'Test whether training on actual model errors improves faithful economic rewriting; detector success and distinctive author transfer are unproven.',
        'selection':'Current expanded epoch-2 outputs only; include unambiguous material errors with locally aligned unchanged references and provisional expression preference.',
        'excluded_policy':'Retain uncertain boundaries and faithful alternatives in the audit, never turn them into rejections automatically.',
        'weighting':'Default proposal is one unit total weight per anchor, divided across its included inputs. Paper weights are recorded as an alternative, not combined automatically.',
        'splits':'All candidates are training-side. Never randomly split sibling inputs across training/validation. Existing development/reserved sets remain unchanged.',
        'future_control':'Same start adapter, candidate prompts, chosen-target exposure and grouped weighting for chosen-only SFT; optimizer/compute controls must be declared before interpreting an objective comparison.',
        'next_gate':'Assess style/content/length contrast and complete CPU tokenizer/mask/response-scoring preflights before proposing a bounded preference pilot.',
        'success':'On unchanged development cases, fewer material issues, expression wins exceeding losses, no increase in exact copying or broken outputs; sparse detector endpoint only on a frozen checkpoint.',
        'gpu_training_authorized':False,'preference_algorithm_selected':False}
    output.mkdir(parents=True);write_rows(output/'preferences.private.jsonl',pairs);write_rows(output/'sft_control.private.jsonl',sft)
    write_rows(output/'decisions.private.jsonl',decisions);write_rows(output/'reference_alignment.private.jsonl',alignment)
    write_rows(output/'diagnostics.private.jsonl',diagnostics);write_json(output/'groups.private.json',{'anchors':dict(anchor_counts),'papers':dict(paper_counts)})
    write_json(output/'summary.json',summary);write_json(output/'protocol.json',protocol)
    write_json(output/'provenance.json',{'model_id':config['model_id'],'model_revision':config['model_revision'],'adapter_sha256':config['adapter_sha256'],
        'decoding':config['decoding'],'seed':config['seed'],'generation_source':'oliverlds/semantic-anchor-4b/1',
        'generation_stage':STAGE,'source_kinds':['original','variant'],'all_targets_seen_by_sft_adapter':True,'human_reference_targets_unchanged':True})
    write_json(output/'receipt.json',{'training_manifest_sha256':file_sha(training/'manifest.json'),'training_payload_sha256':file_sha(training/'training.private.jsonl'),
        'anchor_manifest_sha256':file_sha(anchors/'manifest.json'),'inference_manifest_sha256':file_sha(packet/'manifest.json'),
        'verified_inference_review_manifest_sha256':file_sha(prior_review/'manifest.json'),'raw_inference_report_sha256':file_sha(download/'semantic_anchor_4b_v1/evaluation.json'),
        'bound_source_first_review_sha256':file_sha(bound_review),'metrics_code_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py'))})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','anchors','packet','download','prior-review','bound-review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.training,a.anchors,a.packet,a.download,a.prior_review,a.bound_review,a.output)))


if __name__=='__main__':main()
