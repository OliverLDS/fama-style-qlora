"""Freeze provisional multi-author pairs, detector controls and base-model jobs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from .author_control_replication import verify_files
from .human_prose_annotations import author_key
from .human_prose_review import normalized
from .relation_pilot_review import file_manifest, file_sha
from .signal import distribution
from .author_comparison_corpus import words
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='multi_author_milestone_v1'
MODEL_ID='google/gemma-3-1b-it'
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
CONDITIONS=('ordinary','source_structural_prior')
CONFIG={'protocol':PROTOCOL,'model_id':MODEL_ID,'prompt_format':'multi_author_rewrite_v1',
    'conditions':list(CONDITIONS),'model_revision_policy':'resolve_one_commit_at_first_run_record_and_reuse_for_adapter_comparison',
    'quantization':'NF4 double quantization; FP32 compute','seed':42,
    'decoding':{'do_sample':False,'max_new_tokens':512,'context_limit':4096},
    'structural_prior':'source_only_optional_one_short_pair_plus_training_reference_numeric_profile',
    'profile_fit_split':'training','primary_detector':'gptzero_manual_external',
    'primary_detectability_endpoint':'AI_flag_rate_at_fixed_provider_threshold; baseline_to_adapter_paired_change',
    'detector_label_policy':'AI/Mixed/AI Paraphrased count as flagged; Human unflagged; unknown labels remain unmapped',
    'detector_feasibility_screen':{'complete_matched_pairs':6,'max_human_flags':1,'min_GPT6_rewrite_flags':3,
        'meaning':'small_pilot_screen_not_a_population_accuracy_estimate'},
    'detector_scores_secondary':True,'detector_tuning_or_reward':False,
    'controls_before_gpu':'six_matched_training_human_and_GPT6_rewrite_pairs',
    'minimum_detector_controls':12,'human_control_false_positives_reported':True,
    'detector_missing_statuses':['too_short','unavailable','provider_error','not_tested'],
    'detector_floor_policy':'uninformative_if_controls_do_not_support_origin_discrimination; no_cherry_picking_or_retries',
    'provisional_pilot_progress_gate':{'detectability':'lower_flag_rate_than_prompted_base_at_fixed_settings',
        'meaning':'material_shift_rate_no_increase; local_shifts_separate',
        'readability':'blinded_wins_exceed_losses','copying':'exact_input_copy_rate_no_increase'},
    'reserved_test_policy':'metadata_counts_only_no_text_read_or_scoring',
    'future_conditions':['pooled_qlora_ordinary','plan_conditioned_qlora','author_conditioned_separate_corpus'],
    'training_authorized':False,'gpu_execution_authorized':False}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def stable(text):return sha(PROTOCOL+'|'+text)


def nonreserved_rows(audit):
    # Excluded bodies are never deserialized. Hash verification remains an integrity check.
    with (audit/'audits.private.jsonl').open() as stream:
        lines=[line for line in stream if re.search(r'"split"\s*:\s*"(?:training|development)"',line)]
    return [r for r in map(json.loads,lines) if r['split'] in ('training','development')]


def select_pairs(rows,registry,spotcheck):
    papers={p['paper_id']:p for p in registry['papers']}
    known={r['paragraph_id'] for r in spotcheck['records'] if r['assessment']=='visible_epistemic_strength_change'}
    selected=[];quarantine=[];seen=set();source_hashes=set();target_hashes=set()
    for r in rows:
        if r['split'] not in ('training','development'):raise ValueError('reserved_or_unknown_row')
        p=papers[r['paper_id']]
        if p['split']!=r['split']:raise ValueError('paper_split_mismatch')
        if type(p['publication_year']) is not int or not 1800<=p['publication_year']<2020:raise ValueError('publication_cutoff')
        if r['paragraph_id'] in seen:raise ValueError('duplicate_paragraph')
        seen.add(r['paragraph_id']);a=r['annotation'];reasons=[]
        if not r['schema_valid'] or r['audit']['errors'] or r['audit']['program_errors']:reasons.append('mechanical_error')
        if not a.get('plain_english') or not a.get('source_schema') or a['rewrite_status']=='pending_review' or a['source_quality']['status']=='unusable':reasons.append('unavailable_source')
        if any(c['alignment_status']!='preserved' for c in a['semantic_ledger']['claims']):reasons.append('nonpreserved_claim_annotation')
        if r['paragraph_id'] in known:reasons.append('known_development_epistemic_shift')
        if a.get('plain_english') and normalized(a['plain_english'])==normalized(a['original_paragraph']):reasons.append('identity_pair')
        source_hash=sha(normalized(a.get('plain_english') or ''));target_hash=sha(normalized(a['original_paragraph']))
        if source_hash in source_hashes or target_hash in target_hashes:reasons.append('duplicate_source_or_target')
        if not reasons:
            ss=[s['exact_text'] for s in a['source_schema']['sentences']]
            if normalized(' '.join(ss))!=normalized(a['plain_english']):raise ValueError('source_sentence_coverage')
        if reasons:
            quarantine.append({'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],'split':r['split'],'reasons':reasons,
                'source_sha256':sha(a.get('plain_english') or ''),'target_sha256':sha(a['original_paragraph'])})
            continue
        authors=p['authors'];group='AG_'+stable('|'.join(sorted(author_key(n) for n in authors)))[:12]
        source_hashes.add(source_hash);target_hashes.add(target_hash)
        selected.append({'example_id':'MA_'+stable(r['paragraph_id'])[:16],
            'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],'split':r['split'],
            'authors':authors,'author_group_id':group,'title':p['title'],
            'publication_year':p['publication_year'],'version_year':p.get('version_year'),
            'pdf_sha256':p['pdf_sha256'],'publication_evidence_url':p['publication_evidence_url'],
            'source':a['plain_english'],'target':a['original_paragraph'],'source_sentences':ss,
            'source_sha256':sha(a['plain_english']),'target_sha256':sha(a['original_paragraph']),
            'source_quality_status':a['source_quality']['status'],'needs_review':a['needs_review'],
            'needs_external_context':a['source_quality']['needs_external_context'],
            'data_status':'provisional_not_approved_for_training','annotation_origin':'paired_GPT6_not_independent',
            'prompt_format':CONFIG['prompt_format']})
    training={author_key(n) for r in selected if r['split']=='training' for n in r['authors']}
    development={author_key(n) for r in selected if r['split']=='development' for n in r['authors']}
    if training & development:raise ValueError('author_crosses_splits')
    return selected,quarantine


def profile(training):
    if not training or any(r['split']!='training' for r in training):raise ValueError('training_profile_only')
    sentence_words=[];semicolons=[]
    for r in training:
        # Punctuation approximation is a style proxy, not a syntactic analysis.
        parts=re.split(r'(?<=[.!?])\s+(?=[A-Z])',r['target'])
        sentence_words.extend(len(words(s)) for s in parts)
        semicolons.append(100*r['target'].count(';')/max(1,len(words(r['target']))))
    return {'fit_split':'training','rows':len(training),
        'reference_sentence_words':distribution(sentence_words),
        'reference_semicolons_per_100_words':distribution(semicolons),
        'limitations':'ASCII regex words and heuristic sentence splits; descriptive pooled proxies, not author signature or mandatory targets'}


def structural_prior(row,style_profile):
    pairs=[(len(words(a))+len(words(b)),i+1) for i,(a,b) in enumerate(zip(row['source_sentences'],row['source_sentences'][1:]))]
    eligible=[(n,i) for n,i in pairs if n<=30]
    chosen=min(eligible) if eligible else None
    d=style_profile['reference_sentence_words'];s=style_profile['reference_semicolons_per_100_words']
    return {'representation_version':'source_structural_prior_v1',
        'paragraph_structure':{'operation':'retain_existing_reasoning_order'},
        'sentence_structure':{'operation':'consider_one_merge' if chosen else 'keep_current_groups',
            'source_sentence_numbers':[chosen[1],chosen[1]+1] if chosen else [],
            'reference_sentence_words_iqr':[d['p25'],d['p75']]},
        'language_features':{'reference_semicolons_per_100_words_iqr':[s['p25'],s['p75']],
            'numeric_profile_is_descriptive':True},
        'policy':'Optional only: keep separation if clearer or safer; do not chase numeric rates or chain merges. No vocabulary or semantic relation is supplied.'}


def make_job(row,style_profile,condition):
    if row['split']!='development' or condition not in CONDITIONS:raise ValueError('development_condition_required')
    plan=structural_prior(row,style_profile) if condition=='source_structural_prior' else None
    prompt=INSTRUCTION
    if plan:
        prompt+='\n\nOptional source-only structural prior (use only where it improves clarity):\n'+json.dumps(plan,sort_keys=True)
        prompt+='\nSource sentences:\n'+'\n'.join(str(i)+': '+s for i,s in enumerate(row['source_sentences'],1))
    prompt+='\n\nPassage:\n'+row['source']
    return {'job_id':'JOB_'+stable(row['example_id']+condition)[:16],'example_id':row['example_id'],
        'condition':condition,'source':row['source'],'prompt':prompt,'prompt_sha256':sha(prompt)}


def build(audit,registry_path,spotcheck_path,output):
    verify_files(audit);registry=json.loads(registry_path.read_text());spotcheck=json.loads(spotcheck_path.read_text())
    rows=nonreserved_rows(audit);pairs,quarantine=select_pairs(rows,registry,spotcheck)
    train=[r for r in pairs if r['split']=='training'];dev=[r for r in pairs if r['split']=='development']
    if not train or not dev:raise ValueError('nonempty_training_and_development_required')
    if len({r['paper_id'] for r in train})<6:raise ValueError('six_training_papers_required_for_detector_controls')
    style=profile(train);fresh(output)
    for folder in ('datasets','upload','evaluation','detector_controls/texts'):(output/folder).mkdir(parents=True)
    write_lines(output/'datasets/train.provisional.private.jsonl',[{**r,'prompt':INSTRUCTION+'\n\nPassage:\n'+r['source']} for r in train])
    write_lines(output/'datasets/development.private.jsonl',dev);write_lines(output/'datasets/quarantine.private.jsonl',quarantine)
    write_json(output/'training_style_profile.json',style)
    jobs=sorted([make_job(r,style,c) for r in dev for c in CONDITIONS],key=lambda j:stable(j['job_id']))
    job_path=output/'upload/multi_author_base_jobs.jsonl';write_lines(job_path,jobs)
    config={**CONFIG,'jobs_sha256':file_sha(job_path),'jobs':len(jobs),'development_examples':len(dev)}
    write_json(output/'upload/multi_author_baseline_config.json',config)
    write_json(output/'protocol.json',CONFIG)
    write_lines(output/'evaluation/reference_key.private.jsonl',[{k:r[k] for k in ('example_id','paragraph_id','paper_id','author_group_id','authors','source','target','source_sha256','target_sha256')} for r in dev])
    # Origin calibration uses six training papers, never development or reserved data.
    bypaper={p:[r for r in train if r['paper_id']==p] for p in {r['paper_id'] for r in train}}
    paper_order=sorted(bypaper,key=stable)[:6]
    if len(paper_order)!=6:raise ValueError('six_training_papers_required_for_detector_controls')
    controls=[];key=[]
    for paper in paper_order:
        r=max(bypaper[paper],key=lambda r:(min(len(r['source']),len(r['target'])),stable(r['example_id'])))
        for origin,field in (('historical_human','target'),('GPT6_rewrite','source')):
            cid='DC_'+stable(r['example_id']+origin)[:16];text=r[field]
            controls.append({'item_id':cid,'text':text,'text_sha256':sha(text)})
            key.append({'item_id':cid,'example_id':r['example_id'],'paper_id':paper,'origin':origin,'text_sha256':sha(text)})
            (output/'detector_controls/texts'/('item_'+cid+'.txt')).write_text(text+'\n')
    controls.sort(key=lambda r:stable(r['item_id']));write_lines(output/'detector_controls/items.private.jsonl',controls)
    write_lines(output/'detector_controls/PRIVATE_KEY.jsonl',key)
    with (output/'detector_controls/results_template.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['item_id','text_sha256','provider','test_date_utc','provider_version','threshold_rule','status','provider_label','ai_score','score_scale','notes'])
        writer.writeheader()
        for row in controls:writer.writerow({'item_id':row['item_id'],'text_sha256':row['text_sha256'],'provider':'GPTZero','status':'not_tested'})
    (output/'detector_controls/INSTRUCTIONS.md').write_text('''# Check measurement feasibility first

Use the 12 item text files unchanged in GPTZero if access/quota permits. They are
six matched historical-human/GPT-6-rewrite controls from training papers; IDs hide
origin. Keep PRIVATE_KEY offline. Record actual provider label/score, score scale,
provider version if shown, UTC date and classification rule in results_template.csv.
Use status scored, too_short, unavailable or provider_error; do not silently drop
items. If version or threshold is unknown, say unknown. Never invent scores.
Do not edit, combine, truncate, rerun until a preferred result or upload the whole
package. No credentials go in these files. No calls were submitted by preparation.

Return the filled CSV to Codex for calibration analysis before model execution.
If controls have high false positives or no useful origin separation, the detector
endpoint is uninformative for this corpus. Do not cherry-pick inputs to repair it.
This small check cannot validate a detector generally. Scores do not prove authorship.
''')
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','training_candidates':len(train),'development_examples':len(dev),
        'quarantined':len(quarantine),'quarantine_reasons':dict(Counter(reason for r in quarantine for reason in r['reasons'])),
        'training_papers':len({r['paper_id'] for r in train}),'development_papers':len({r['paper_id'] for r in dev}),
        'training_author_groups':len({r['author_group_id'] for r in train}),
        'reserved_metadata_only':{'papers':sum(p['split']=='reserved_test' for p in registry['papers']),'expected_paragraphs':5*sum(p['split']=='reserved_test' for p in registry['papers'])},
        'baseline_jobs':len(jobs),'conditions':list(CONDITIONS),'detector_controls':len(controls),
        'training_candidates_needing_review':sum(r['needs_review'] for r in train),
        'development_examples_needing_review':sum(r['needs_review'] for r in dev),
        'reference_text_exported_to_generator':False,'author_labels_exported_to_generator':False,'reference_plans_exported':False,
        'training_approved':False,'detector_calls':0,'new_model_calls':0,'gpu_jobs_started':0,'reserved_content_used':False}
    write_json(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('''# Multi-author milestone: measurement and baseline preparation

First read docs/DEVELOPMENT_PLAN.md and docs/multi_author_v1/PROTOCOL.md in the repo.
This packet contains provisional training candidates, not approved training data.
Reserved examples remain sealed. No model/detector/GPU run has been started.

1. Begin with detector_controls/INSTRUCTIONS.md and the 12 fixed item texts.
   If you choose to test them in GPTZero, fill results_template.csv unchanged IDs,
   retain actual labels/scales and report missing/too-short cases. Return the CSV
   to Codex. Never upload PRIVATE_KEY, references, training files or this package.
2. The prompted-base notebook is prepared at kaggle/multi_author_baseline.ipynb.
   Wait for explicit model-execution authorization. After authorization, upload
   ONLY the two JSON files in upload/ as a private Kaggle Dataset, attach that
   input, enable HF_TOKEN in Secrets, and select T4 x2. The notebook masks GPU 0
   and performs inference only. Do not use Kaggle CLI push or start training.
3. Return baseline outputs for evidence/copying/readability evaluation and frozen
   detector export. Do not edit or regenerate outputs to improve detector scores.

The structural arm is a source-only prior plus pooled training-reference numeric
features, not a learned taxonomy mapper or an oracle author plan. It may fail.
Keep ordinary rewriting and unchanged/source/human controls in the comparison.
Any future QLoRA pilot needs separate authorization and exact tokenizer/label
checks. No full training is authorized here. Preserve unknown effective effort
and possible chat retrieval as limitations; do not claim independent verification.
''')
    file_manifest(output,protocol=PROTOCOL,code_sha256=file_sha(Path(__file__)),
        upstream_audit_manifest_sha256=file_sha(audit/'manifest.json'),registry_sha256=file_sha(registry_path),spotcheck_sha256=file_sha(spotcheck_path))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','registry','spotcheck','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.audit,a.registry,a.spotcheck,a.output),indent=2))


if __name__=='__main__':main()
