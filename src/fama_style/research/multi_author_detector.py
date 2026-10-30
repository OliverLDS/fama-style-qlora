"""Record an exported detector result and freeze a sparse, unpaid evaluation plan.

Consumes extracted page-one text, never contacts a detector or creates accounts.
"""
import argparse
import csv
from datetime import date
import json
from pathlib import Path
import re

from .author_control_replication import verify_files
from .relation_pilot_review import file_manifest, file_sha, read_rows
from .multi_author_milestone import sha
from fama_style.rhetoric.experiment import fresh, write_json

PROTOCOL='multi_author_sparse_detector_v2'
CONFIG={'protocol':PROTOCOL,'provider':'GPTZero','word_allowance_user_reported':10000,
    'target_max_project_words':8000,'reserve_words':2000,'paid_accounts':False,
    'additional_accounts_required':False,'use':'sparse_frozen_checkpoint_only_not_epoch_or_prompt_selection',
    'calibration_next':'matched_human_control_only; remaining_ten_not_requested_now',
    'six_pair_screen_status':'incomplete_v1_diagnostic_not_a_new_execution_prerequisite',
    'raw_rewrite_floor_does_not_establish_Gemma_output_floor':True,
    'evaluation_examples':8,'examples_per_development_paper':2,'minimum_source_and_reference_chars':500,
    'subset_selection':'fixed_hash_within_paper_before_model_outputs; round_robin_papers',
    'model_arms':['base_ordinary','base_source_structural_prior','future_pooled_adapter_ordinary'],
    'control_arms':['GPT6_input','historical_human_reference'],
    'output_budget_estimate':'reference_word_count_per_model_arm; replace_with_actual_counts_before_scanning',
    'over_budget_policy':'scan_complete_paper_balanced_rounds_only; first_four_then_eight; no_trimming_or_best_output_selection',
    'previously_scanned_unchanged_texts':'reuse_result_with_provider_date_version; do_not_treat_different_versions_as_matched',
    'local_development':'copying_and_length_checks; automated_claim_and_readability_proxies; target_loss_diagnostic',
    'quota_counter':'whitespace_estimate_not_provider_billing; actual_remaining_allowance_unknown',
    'independence':'same_model_and_retrieval_uncontrolled','detector_scores_are_authorship_proof':False}


def parse_first_page(text,expected):
    normalized=lambda x:' '.join(x.split())
    if normalized(expected) not in normalized(text):raise ValueError('report_text_does_not_match_control')
    label=re.search(r'We predict this text is\s+(Human Generated|AI Generated|Mixed)',text)
    probability=re.search(r'AI Probability\s+(\d+(?:\.\d+)?)\s*%',text)
    version=re.search(r'\bVersion\s+(\d{4}-\d{2}-\d{2}-[a-z0-9._-]+)',text)
    report_date=re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',text)
    if not label or not probability:raise ValueError('unrecognized_report_label_or_probability')
    score=float(probability[1])
    if not 0<=score<=100:raise ValueError('probability_range')
    day=date(int(report_date[3]),int(report_date[1]),int(report_date[2])).isoformat() if report_date else None
    return {'provider_label':label[1],'flagged':label[1]!='Human Generated','ai_probability_percent':score,
        'score_interpretation':'provider document-level AI probability; not percent of words written by AI',
        'provider_version':version[1] if version else None,'report_date':day,'report_timezone':None,
        'numeric_decision_threshold':None,'label_rule':'provider-native categorical label',
        'text_matches_control_apart_from_whitespace':True,'pages_used':[1]}


def sparse_subset(development):
    if any(r['split']!='development' for r in development):raise ValueError('development_only')
    papers=sorted({r['paper_id'] for r in development});groups={}
    if len(papers)!=4:raise ValueError('four_development_papers')
    for paper in papers:
        pool=[r for r in development if r['paper_id']==paper and min(len(r['source']),len(r['target']))>=500]
        groups[paper]=sorted(pool,key=lambda r:sha(PROTOCOL+'|'+r['example_id']))[:2]
        if len(groups[paper])!=2:raise ValueError('two_length_eligible_examples_per_paper')
    return [{**{k:groups[p][i][k] for k in ('example_id','paper_id','source_sha256','target_sha256')},
             'round':i+1,'source_words_estimate':len(groups[p][i]['source'].split()),
             'human_words_estimate':len(groups[p][i]['target'].split())}
            for i in range(2) for p in papers]


def run(package,page_json,pdf,item_id,output):
    verify_files(package)
    items={r['item_id']:r for r in read_rows(package/'detector_controls/items.private.jsonl')}
    key={r['item_id']:r for r in read_rows(package/'detector_controls/PRIVATE_KEY.jsonl')}
    item=items[item_id];page=json.loads(page_json.read_text())
    if page['page_extracted']!=1:raise ValueError('page_one_required')
    parsed=parse_first_page(page['text'],item['text'])
    counterpart=[k for k,r in key.items() if r['example_id']==key[item_id]['example_id'] and k!=item_id]
    if len(counterpart)!=1:raise ValueError('one_matched_control_required')
    counterpart=counterpart[0]
    subset=sparse_subset(read_rows(package/'datasets/development.private.jsonl'))
    source_words=sum(r['source_words_estimate'] for r in subset);human_words=sum(r['human_words_estimate'] for r in subset)
    all_controls=sum(len(r['text'].split()) for r in items.values())
    pair_words=len(item['text'].split())+len(items[counterpart]['text'].split())
    record={'item_id':item_id,'text_sha256':item['text_sha256'],'origin':key[item_id]['origin'],
        **parsed,'report_pdf_sha256':file_sha(pdf),'extracted_page_sha256':file_sha(page_json),
        'words_estimate':len(item['text'].split()),'actual_words_charged':None,
        'input_already_unflagged':not parsed['flagged'],'training_or_adapter_used':False}
    fresh(output);write_json(output/'observation.private.json',record);write_json(output/'protocol.json',CONFIG)
    write_json(output/'PRIVATE_EVALUATION_SUBSET.json',subset)
    template=package/'detector_controls/results_template.csv'
    with template.open(newline='') as f:reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
    for row in rows:
        if row['item_id']==item_id:row.update(provider='GPTZero',provider_version=parsed['provider_version'] or 'unknown',
            threshold_rule='provider-native label; numeric threshold not shown',status='scored',provider_label=parsed['provider_label'],
            ai_score=str(parsed['ai_probability_percent']),score_scale='0-100 document AI probability',
            notes='PDF date '+str(parsed['report_date'])+'; timezone not shown, UTC date left blank. Page 1 only.')
    with (output/'results_working.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={'protocol':PROTOCOL,'controls_recorded':1,'controls_total':len(items),'observed_control_origin':record['origin'],
        'provider_label':parsed['provider_label'],'ai_probability_percent':parsed['ai_probability_percent'],
        'provider_version':parsed['provider_version'],'report_date':parsed['report_date'],'timezone_known':False,
        'input_already_unflagged':not parsed['flagged'],'numeric_threshold_known':False,
        'matched_control_pending':True,'next_control_id':counterpart,'next_control_words_estimate':len(items[counterpart]['text'].split()),
        'all_control_words_estimate':all_controls,'initial_matched_pair_words_estimate':pair_words,
        'evaluation_subset_examples':len(subset),'evaluation_subset_source_words':source_words,'evaluation_subset_human_words':human_words,
        'three_model_arms_words_estimate':3*human_words,
        'checkpoint_with_initial_pair_words_estimate':source_words+4*human_words+pair_words,
        'checkpoint_with_all_controls_words_estimate':source_words+4*human_words+all_controls,
        'model_output_word_counts_observed':False,'detector_accuracy_or_feasibility_established':False,
        'provider_free_allowance_is_user_reported':True,'paid_tools_required':False,'detector_calls_by_agent':0,
        'gpu_jobs_started':0,'training_called':False,'reserved_content_used':False}
    write_json(output/'summary.json',summary)
    (output/'NEXT_STEP.md').write_text('''# Next scan: matched human original only

The supplied GPT-6-rewrite control was labeled Human Generated with 3%% document
AI probability before training. It already has no binary AI flag to reduce.
One observation does not establish corpus-wide detector performance or how
Gemma outputs will score. The base-versus-adapter endpoint is still unmeasured.

Scan detector_controls/texts/item_%s.txt from the original package next.
Leave the other ten controls for now. Record the actual label, score/scale, date
and visible version; a screenshot or PDF is optional. Preserve first results.
Do not edit text or rescan until a preferred result. The 12 original controls
total about %d whitespace words, but full calibration is not requested now.

Use results_working.csv here for subsequent entries; the original packet/template
is immutable. The existing recorded row has no fabricated UTC date or provider
decision threshold. Share only results, never credentials. No paid account or
additional account is needed for the proposed project budget.

GPTZero is now a sparse external checkpoint measurement. Local claim, copying,
readability and loss diagnostics handle frequent development. Eight development
examples are fixed privately before model outputs, two per paper, with source and
reference length at least 500 characters. This is a length-selected subset.
Keep input/human/base/structured/adapter variants paired; freeze all outputs
before scans. Count actual output words and remaining provider allowance first.
If the batch exceeds the project budget, use complete four-paper rounds (four
then eight examples), never trim text or choose the best-looking outputs.

Estimated checkpoint totals use reference lengths for three model arms and are
not provider billing or observed output counts. Include control usage, leave
headroom, reuse prior unchanged results only with explicit date/version caveats,
and do not tune prompts/epochs against detector feedback. The original six-pair
feasibility gate remains an incomplete v1 diagnostic; no pass is claimed from one
pair and all 12 scans are not a new execution prerequisite. A later baseline
requires explicit GPU authorization. The Fama/style objective and reserved splits
remain unchanged.
'''%(counterpart,all_controls))
    file_manifest(output,protocol=PROTOCOL,code_sha256=file_sha(Path(__file__)),
        initial_package_manifest_sha256=file_sha(package/'manifest.json'),report_pdf_sha256=file_sha(pdf),first_page_sha256=file_sha(page_json))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','page-json','pdf','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--item-id',required=True);a=p.parse_args()
    print(json.dumps(run(a.package,a.page_json,a.pdf,a.item_id,a.output),indent=2))


if __name__=='__main__':main()
