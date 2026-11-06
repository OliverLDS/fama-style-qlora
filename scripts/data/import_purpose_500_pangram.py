"""Import matched Gemini-route Pangram PDFs; retain failures and body-only scans."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


packet=load('purpose_Pangram_packet','prepare_purpose_500_pangram.py')
parser=load('complete_Pangram_parser','import_corpus_500_full_pangram.py')
from fama_style.research.detector_provider_comparison import bind

RUN=packet.RUN
RECEIVED=RUN/'pangram_reports_v1'
OUTPUT=RUN/'pangram_import_v1'
PREFIXES=(
    'The following paragraph is a re-expression of the original passage, written in natural, human-written academic economics prose:',
    'Rewritten Paragraph:',
)


def bind_with_prefix_audit(page,cards,pdf_sha,filename):
    try:
        item,method=bind(page,cards,pdf_sha,filename,{})
        return item,method,None
    except ValueError as original_error:
        if str(original_error)!='unique_complete_text_binding_required':raise
        adjusted=[]
        for r in cards:
            if r['condition']!='epoch_1':continue
            for prefix in PREFIXES:
                if r['text'].startswith(prefix+'\n\n'):
                    body=r['text'][len(prefix+'\n\n'):]
                    adjusted.append(dict(r,text=body,text_sha256=packet.review.runtime.sha(body),
                        removed_model_prefix=prefix,full_model_output_sha256=r['text_sha256']))
        if not adjusted:raise original_error
        item,method=bind(page,adjusted,pdf_sha,filename,{})
        audit=dict(scan_id=item['scan_id'],pdf_sha256=pdf_sha,
            deviation='model_preface_or_heading_omitted_from_scan; paragraph body otherwise matched',
            removed_prefix=item['removed_model_prefix'],removed_separator='\n\n',
            full_model_output_sha256=item['full_model_output_sha256'],scanned_body_sha256=item['text_sha256'],
            model_output_edited=False,report_score_modified=False,new_scan_requested=False,
            comparison_limit='Score applies to paragraph body, not exact full model output.')
        return item,'body_only_after_versioned_prefix_audit::'+method,audit


def condition_summary(group):
    return dict(scored_texts=len(group),native_labels=dict(Counter(r['provider_label'] for r in group)),
        entirely_human=sum(r['entirely_human'] for r in group),
        mean_human_written_share=statistics.mean(r['human_written_document_percent'] for r in group),
        median_human_written_share=statistics.median(r['human_written_document_percent'] for r in group),
        mean_AI_generated_share=statistics.mean(r['ai_generated_document_percent'] for r in group),
        mean_AI_assisted_share=statistics.mean(r['ai_assisted_document_percent'] for r in group),
        body_only_reports=sum(not r['full_frozen_text_scanned'] for r in group))


def compare(observations,coverage):
    by={(r['content_id'],r['input_condition'],r['condition']):r for r in observations}
    if len(by)!=len(observations):raise ValueError('duplicate_scan_needs_audit')
    cases=sorted({(r['content_id'],r['input_condition']) for r in coverage})
    if len(coverage)!=24 or len(cases)!=8:raise ValueError('full_intended_panel_required')
    pairs=[]
    for cid,route in cases:
        entries={c:by.get((cid,route,c)) for c in ['input','base','epoch_1']}
        source=entries['input'];base=entries['base'];trained=entries['epoch_1']
        if source is None or base is None:raise ValueError('input_and_base_reports_required')
        pairs.append(dict(content_id=cid,input_condition=route,style_group=source['style_group'],
            conditions={c:None if r is None else {k:r[k] for k in ['provider_label','entirely_human','human_written_document_percent',
                'ai_generated_document_percent','ai_assisted_document_percent','full_frozen_text_scanned','report_date','provider_version']} for c,r in entries.items()},
            adapter_task_failed=trained is None,
            adapter_minus_input_human_share=None if trained is None else trained['human_written_document_percent']-source['human_written_document_percent'],
            adapter_minus_base_human_share=None if trained is None else trained['human_written_document_percent']-base['human_written_document_percent'],
            strict_full_text_triplet=trained is not None and all(r['full_frozen_text_scanned'] for r in entries.values()),
            new_entirely_human_conversion=trained is not None and not source['entirely_human'] and trained['entirely_human']))
    scored=[r for r in pairs if not r['adapter_task_failed']]
    strict=[r for r in scored if r['strict_full_text_triplet']]
    def paired_stats(group):
        return dict(scored_pairs=len(group),higher_human_share_than_input=sum(r['adapter_minus_input_human_share']>0 for r in group),
            lower_human_share_than_input=sum(r['adapter_minus_input_human_share']<0 for r in group),
            same_human_share_as_input=sum(r['adapter_minus_input_human_share']==0 for r in group),
            higher_human_share_than_base=sum(r['adapter_minus_base_human_share']>0 for r in group),
            new_entirely_human_conversions=sum(r['new_entirely_human_conversion'] for r in group),
            AI_related_input_pairs=sum(not r['conditions']['input']['entirely_human'] for r in group))
    return dict(protocol='purpose_500_reexpress_pangram_import_v1',intended_cases=8,underlying_contents=4,
        new_PDF_reports=19,reused_input_reports=4,scored_texts=len(observations),unscanned_task_failures=1,
        conditions={route:{c:condition_summary([r for r in observations if r['input_condition']==route and r['condition']==c])
            for c in ['input','base','epoch_1']} for route in ['gemini_input','fresh_reconstruction']},
        paired=paired_stats(scored),strict_full_text_sensitivity=paired_stats(strict),pairs=pairs,
        provider_versions=sorted({r['provider_version'] for r in observations}),report_dates=sorted({r['report_date'] for r in observations}),
        body_only_reports=sum(not r['full_frozen_text_scanned'] for r in observations),
        exact_prior_inputs_reused=True,published_human_rescans=0,displayed_shares_rescaled=False,
        score_semantics='Native document authorship shares and labels, not document AI probabilities.',
        detector_calls_by_agent=0,new_model_calls=0,new_GPU_jobs=0,
        limits=['Four reused development contents across two Gemini routes, not eight independent contents or a sealed test.',
            'Four exact historical input scores are reused across a one-day date difference, with the same displayed Pangram version.',
            'Two PDF reports omit model-generated heading/prefatory text. Scores apply to bodies; strict sensitivity excludes those pairs.',
            'One short missing-passage response remains a task failure with unavailable detector score, not a zero score or human-label success.',
            'A 0 versus 0.1 display-floor change with the same AI native label is not persuasive detector improvement.',
            'Meaning/readability findings remain separate from detector labels; Fama-specific style is not established.',
            'Per-report account identity and identical provider configurations have not been independently verified.'])


def build():
    if OUTPUT.exists():raise ValueError('refuse_overwrite_Pangram_import')
    packet.verify_files(packet.OUTPUT);packet.verify_files(RECEIVED);packet.prior_observations()
    cards=packet.review.rows(packet.OUTPUT/'PRIVATE_KEY.jsonl')
    by_id={r['scan_id']:r for r in cards};expected={r['scan_id'] for r in cards if r['status']=='new_scan_required'}
    observations=[];coverage=[];deviations=[];seen=set()
    for entry in packet.review.read(RECEIVED/'receipt.private.json'):
        h=entry['pdf_sha256'];pdf=RECEIVED/'raw_pdfs'/(h+'.pdf');ex=RECEIVED/'extracted'/(h+'.json')
        if packet.review.runtime.file_sha(pdf)!=h or packet.review.runtime.file_sha(ex)!=entry['extracted_sha256']:raise ValueError('PDF_evidence_changed')
        page=packet.review.read(ex)
        item,method,audit=bind_with_prefix_audit(page,[r for r in cards if r['status']=='new_scan_required'],h,Path(entry['source']).name)
        if item['scan_id'] in seen:raise ValueError('duplicate_report_needs_audit')
        seen.add(item['scan_id']);original=by_id[item['scan_id']]
        if audit:deviations.append(audit)
        observations.append(dict(**{k:original[k] for k in ['scan_id','content_id','input_condition','condition','style_group','paper_id']},
            frozen_text_sha256=original['text_sha256'],scanned_text_sha256=item['text_sha256'],full_frozen_text_scanned=audit is None,
            evidence_origin='new_PDF',pdf_sha256=h,binding_method=method,
            **parser.parse_page(page['text'])))
    if seen!=expected:raise ValueError('missing_or_unexpected_report')
    for item in cards:
        if item['status']=='reused_exact_prior_observation':
            old=item['prior_observation']
            if old['text_sha256']!=item['text_sha256']:raise ValueError('prior_text_changed')
            observations.append(dict(old,**{k:item[k] for k in ['scan_id','content_id','input_condition','condition','style_group','paper_id']},
                frozen_text_sha256=item['text_sha256'],scanned_text_sha256=item['text_sha256'],full_frozen_text_scanned=True,
                evidence_origin='exact_prior_input_reuse'))
        coverage.append(dict(**{k:item[k] for k in ['scan_id','content_id','input_condition','condition','style_group','text_sha256']},
            status='task_failed_no_detector_score' if item['status']=='not_scanned_task_failure' else 'scored',
            failure_note=item['failure_note']))
    summary=compare(observations,coverage)
    OUTPUT.mkdir()
    for name,values in [('observations.private.jsonl',observations),('panel_coverage.private.jsonl',coverage),
        ('protocol_deviations.private.jsonl',deviations),('paired_results.private.jsonl',summary['pairs'])]:
        packet.review.write_rows(OUTPUT/name,values)
    packet.review.runtime.write_json(OUTPUT/'summary.json',summary)
    packet.review.runtime.write_json(OUTPUT/'manifest.json',dict(builder_sha256=packet.review.runtime.file_sha(Path(__file__)),
        scan_manifest_sha256=packet.review.runtime.file_sha(packet.OUTPUT/'manifest.json'),
        received_manifest_sha256=packet.review.runtime.file_sha(RECEIVED/'manifest.json'),
        prior_full_import_manifest_sha256=packet.review.runtime.file_sha(packet.HISTORY/'pangram_full_import_v1/manifest.json'),
        parser_sha256=packet.review.runtime.file_sha(Path(parser.__file__)),
        binding_sha256=packet.review.runtime.file_sha(ROOT/'src/fama_style/research/detector_provider_comparison.py'),
        extractor_sha256=packet.review.runtime.file_sha(ROOT/'scripts/data/extract_detector_pdf_page.swift'),
        files={str(p.relative_to(OUTPUT)):packet.review.runtime.file_sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    result=build();print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
