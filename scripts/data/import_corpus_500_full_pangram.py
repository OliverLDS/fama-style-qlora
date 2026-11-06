"""Import the remaining PDFs and freeze the complete eight-case Pangram panel."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('initial500',Path(__file__).with_name('import_corpus_500_pangram.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
EXTRA_SUMMARIES={
    'We believe this text is mainly AI, with some human-written content.':'Mainly AI; some human',
    'We believe that this text is a mix of AI, AI-assisted, and human-written content.':'Mix: AI / AI-assisted / human',
}


def parse_page(text):
    summaries=re.findall(r'We believe[^\n]+',text)
    if len(summaries)!=1:raise ValueError('unique_provider_summary_required')
    summary=summaries[0]
    if summary not in EXTRA_SUMMARIES:return m.parser.parse(text)
    # Reuse the strict existing numeric/layout parser, then restore the exact
    # newly observed native summary. No score or percentage is inferred.
    known='We believe that this text is a mix of AI and human-written content.'
    result=m.parser.parse(text.replace(summary,known,1))
    result.update(provider_summary=summary,provider_label=EXTRA_SUMMARIES[summary],entirely_human=False)
    return result


def verify_files(folder):
    manifest=m.read(folder/'manifest.json')
    for name,h in manifest['files'].items():
        if m.file_sha(folder/name)!=h:raise ValueError('frozen_artifact_changed')
    return manifest


def compare(observations,expected_ids):
    if len(expected_ids)!=8 or len(set(expected_ids))!=8:raise ValueError('eight_frozen_cases_required')
    by={(x['example_id'],x['condition']):x for x in observations}
    if len(by)!=len(observations):raise ValueError('duplicate_report_requires_audit')
    expected={(rid,c) for rid in expected_ids for c in ['input','base','epoch_1']}
    if set(by)!=expected:raise ValueError('all_twenty_four_reports_required')
    pairs=[]
    for rid in expected_ids:
        source,base,trained=[by[rid,c] for c in ['input','base','epoch_1']]
        if any(source[k]!=x[k] for x in [base,trained] for k in ['paper_id','style_group']):raise ValueError('triplet_lineage_changed')
        pairs.append(dict(example_id=rid,style_group=source['style_group'],paper_id=source['paper_id'],
            conditions={c:{k:by[rid,c][k] for k in ['provider_label','human_written_document_percent','ai_generated_document_percent','ai_assisted_document_percent','provider_word_count','material_output_issue_identified','source_word_sequence_overlap']} for c in ['input','base','epoch_1']},
            change_human_share_vs_input=trained['human_written_document_percent']-source['human_written_document_percent'],
            change_human_share_vs_base=trained['human_written_document_percent']-base['human_written_document_percent'],
            AI_related_input_becomes_entirely_human=not source['entirely_human'] and trained['entirely_human'],
            entirely_human_input_stays_entirely_human=source['entirely_human'] and trained['entirely_human']))
    conditions={}
    for c in ['input','base','epoch_1']:
        group=[by[rid,c] for rid in expected_ids]
        conditions[c]=dict(n=8,native_labels=dict(Counter(x['provider_label'] for x in group)),
            entirely_human=sum(x['entirely_human'] for x in group),
            mean_human_share=statistics.mean(x['human_written_document_percent'] for x in group),
            median_human_share=statistics.median(x['human_written_document_percent'] for x in group),
            mean_AI_generated_share=statistics.mean(x['ai_generated_document_percent'] for x in group),
            mean_AI_assisted_share=statistics.mean(x['ai_assisted_document_percent'] for x in group),
            median_source_overlap=statistics.median(x['source_word_sequence_overlap'] for x in group),
            material_output_issues_identified=None if c=='input' else sum(x['material_output_issue_identified'] for x in group),
            entirely_human_outputs_with_material_issues=None if c=='input' else sum(x['entirely_human'] and x['material_output_issue_identified'] for x in group))
    return dict(cases=8,reports=24,pairs=pairs,conditions=conditions,
        higher_human_share_than_input=sum(x['change_human_share_vs_input']>0 for x in pairs),
        lower_human_share_than_input=sum(x['change_human_share_vs_input']<0 for x in pairs),
        unchanged_human_share_vs_input=sum(x['change_human_share_vs_input']==0 for x in pairs),
        higher_human_share_than_base=sum(x['change_human_share_vs_base']>0 for x in pairs),
        lower_human_share_than_base=sum(x['change_human_share_vs_base']<0 for x in pairs),
        AI_related_inputs_available=sum(not by[rid,'input']['entirely_human'] for rid in expected_ids),
        AI_related_inputs_becoming_entirely_human=sum(x['AI_related_input_becomes_entirely_human'] for x in pairs),
        entirely_human_inputs_staying_entirely_human=sum(x['entirely_human_input_stays_entirely_human'] for x in pairs),
        provider_versions=sorted({x['provider_version'] for x in observations}),
        report_dates=sorted({x['report_date'] for x in observations}),
        binding_methods=dict(Counter(x['binding_method'] for x in observations)),
        displayed_shares_rescaled=False,score_semantics='Displayed document authorship shares, not document AI probabilities.',
        new_detector_calls=0,new_model_calls=0,new_GPU_jobs=0,
        limits=['Eight reused development contents, two per author group; not a population estimate or sealed test.',
            'The first two reports and later six cases use the same preselected panel; no post-score case exclusion.',
            'Native AI Generated and AI Assisted labels are distinct; neither alone is Human Written.',
            'Tiny displayed percentage floors are retained even when shares sum above 100.',
            'Some mathematical outputs contain literal HTML subscripts; format and input retention remain confounds.',
            'Meaning judgments are provisional same-agent assessments, not calibrated ground truth.',
            'Account identity and settings equivalence are not independently verified.'])


def build(run,packet,output):
    if output.exists():raise ValueError('refuse_overwrite_full_import')
    verify_files(run/'review_v1');verify_files(packet)
    original=run/'pangram_import_v1';old_manifest=verify_files(original)
    for name,h in old_manifest['evidence'].items():
        if m.file_sha(run/'pangram_reports_v1'/name)!=h:raise ValueError('initial_PDF_evidence_changed')
    if m.file_sha(run/'pangram_reports_v1/receipt.private.json')!=old_manifest['receipt_sha256']:raise ValueError('initial_receipt_changed')
    config=m.read(packet/'upload/corpus_500_config.json');expected_ids=config['detector_case_ids']
    scans=run/'review_v1/pangram_remaining_v1';verify_files(scans)
    if m.read(scans/'manifest.json')['frozen_config_sha256']!=m.file_sha(packet/'upload/corpus_500_config.json'):raise ValueError('frozen_panel_changed')
    items=m.rows(scans/'PRIVATE_KEY.jsonl')
    if len(items)!=18:raise ValueError('eighteen_remaining_texts_required')
    initial=m.rows(original/'observations.private.jsonl')
    if len(initial)!=6 or {x['scan_id'] for x in initial}&{x['scan_id'] for x in items}:raise ValueError('initial_remainder_overlap')
    outputs={(x['example_id'],x['snapshot']):x for x in m.rows(run/'review_v1/outputs_and_assessments.private.jsonl')}
    received=run/'pangram_remaining_reports_v1';new=[];seen=set();evidence={}
    for entry in m.read(received/'receipt.private.json'):
        h=entry['pdf_sha256'];pdf=received/'raw_pdfs'/(h+'.pdf');ex=received/'extracted'/(h+'.json')
        if m.file_sha(pdf)!=h or m.file_sha(ex)!=entry['extracted_sha256']:raise ValueError('remaining_PDF_evidence_changed')
        page=m.read(ex);item,method=m.bind(page,items,h,Path(entry['source']).name,{})
        if item['scan_id'] in seen:raise ValueError('duplicate_remaining_report')
        seen.add(item['scan_id'])
        if m.file_sha(scans/'texts'/(item['scan_id']+'.txt'))!=item['text_sha256']:raise ValueError('frozen_scan_text_changed')
        source=outputs[item['example_id'],'base' if item['condition']=='input' else item['condition']]
        text=source['source' if item['condition']=='input' else 'output']
        if hashlib.sha256(text.encode()).hexdigest()!=item['text_sha256']:raise ValueError('review_text_changed')
        new.append(dict(**{k:item[k] for k in ['scan_id','example_id','condition','text_sha256']},paper_id=source['paper_id'],style_group=source['style_group'],
            pdf_sha256=h,binding_method=method,
            material_output_issue_identified=None if item['condition']=='input' else source['assessment']['material_claim_change'],
            source_word_sequence_overlap=1.0 if item['condition']=='input' else source['source_metrics']['word_sequence_overlap'],**parse_page(page['text'])))
        evidence[str(pdf.relative_to(received))]=h;evidence[str(ex.relative_to(received))]=entry['extracted_sha256']
    if seen!={x['scan_id'] for x in items}:raise ValueError('remaining_report_coverage')
    summary=compare(initial+new,expected_ids);summary['protocol']='corpus_500_pangram_full_v1'
    output.mkdir();(output/'observations.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in initial+new))
    m.write(output/'summary.json',summary)
    m.write(output/'manifest.json',dict(builder_sha256=m.file_sha(Path(__file__)),initial_import_manifest_sha256=m.file_sha(original/'manifest.json'),
        remaining_receipt_sha256=m.file_sha(received/'receipt.private.json'),remaining_scan_manifest_sha256=m.file_sha(scans/'manifest.json'),
        previous_parser_sha256=m.file_sha(Path(m.parser.__file__)),binding_sha256=m.file_sha(ROOT/'src/fama_style/research/detector_provider_comparison.py'),
        evidence=evidence,files={p.name:m.file_sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');p.add_argument('--packet',type=Path,default=ROOT/'data/corpus_500_pilot_v1');a=p.parse_args()
    result=build(a.run,a.packet,a.run/'pangram_full_import_v1')
    print(json.dumps({k:result[k] for k in result if k!='pairs'},indent=2))
