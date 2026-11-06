"""Import reference controls without repeating or changing prior detector results."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('full500',Path(__file__).with_name('import_corpus_500_full_pangram.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


def combine(references,full):
    by={x['example_id']:x for x in references};ids=[x['example_id'] for x in full['pairs']]
    if len(references)!=8 or len(by)!=8 or set(by)!=set(ids):raise ValueError('eight_matched_references_required')
    pairs=[]
    for prior in full['pairs']:
        ref=by[prior['example_id']]
        if any(ref[k]!=prior[k] for k in ['paper_id','style_group']):raise ValueError('reference_lineage_changed')
        pairs.append(dict(prior,reference={k:ref[k] for k in ['provider_label','human_written_document_percent','ai_generated_document_percent','ai_assisted_document_percent','text_sha256']},
            trained_human_share_minus_reference=prior['conditions']['epoch_1']['human_written_document_percent']-ref['human_written_document_percent']))
    return dict(protocol='corpus_500_reference_pangram_v1',reference_reports=8,combined_reports=32,
        human_reference_native_labels=dict(Counter(x['provider_label'] for x in references)),
        human_reference_entirely_human=sum(x['entirely_human'] for x in references),
        human_reference_human_shares=[by[rid]['human_written_document_percent'] for rid in ids],
        prior_model_results_unchanged=True,prior_conditions=full['conditions'],pairs=pairs,
        score_semantics=full['score_semantics'],displayed_shares_rescaled=False,
        new_detector_calls=0,new_model_calls=0,new_GPU_jobs=0,
        interpretation='Reference controls complete; previous matched input/output findings unchanged. Routine additional published-reference scans are not requested.')


def build(run,output):
    if output.exists():raise ValueError('refuse_overwrite_reference_import')
    packet=run/'review_v1/pangram_reference_controls_v1';m.verify_files(packet)
    full=run/'pangram_full_import_v1';m.verify_files(full)
    if m.m.read(packet/'manifest.json')['full_panel_manifest_sha256']!=m.m.file_sha(full/'manifest.json'):raise ValueError('parent_panel_changed')
    items=m.m.rows(packet/'PRIVATE_KEY.jsonl');received=run/'pangram_reference_reports_v1';observations=[];evidence={};seen=set()
    for entry in m.m.read(received/'receipt.private.json'):
        h=entry['pdf_sha256'];pdf=received/'raw_pdfs'/(h+'.pdf');ex=received/'extracted'/(h+'.json')
        if m.m.file_sha(pdf)!=h or m.m.file_sha(ex)!=entry['extracted_sha256']:raise ValueError('reference_PDF_evidence_changed')
        page=m.m.read(ex);item,method=m.m.bind(page,items,h,Path(entry['source']).name,{})
        if item['scan_id'] in seen:raise ValueError('duplicate_reference_report')
        seen.add(item['scan_id'])
        if m.m.file_sha(packet/'texts'/(item['scan_id']+'.txt'))!=item['text_sha256']:raise ValueError('reference_text_changed')
        observations.append(dict(**{k:item[k] for k in ['scan_id','example_id','paper_id','style_group','condition','text_sha256']},pdf_sha256=h,binding_method=method,**m.parse_page(page['text'])))
        evidence[str(pdf.relative_to(received))]=h;evidence[str(ex.relative_to(received))]=entry['extracted_sha256']
    if seen!={x['scan_id'] for x in items}:raise ValueError('reference_report_coverage')
    summary=combine(observations,m.m.read(full/'summary.json'))
    summary.update(provider_versions=sorted({x['provider_version'] for x in observations}),report_dates=sorted({x['report_date'] for x in observations}),binding_methods=dict(Counter(x['binding_method'] for x in observations)))
    output.mkdir();(output/'observations.private.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in observations));m.m.write(output/'summary.json',summary)
    m.m.write(output/'manifest.json',dict(builder_sha256=m.m.file_sha(Path(__file__)),receipt_sha256=m.m.file_sha(received/'receipt.private.json'),reference_packet_manifest_sha256=m.m.file_sha(packet/'manifest.json'),parent_panel_manifest_sha256=m.m.file_sha(full/'manifest.json'),evidence=evidence,files={p.name:m.m.file_sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');a=p.parse_args();summary=build(a.run,a.run/'pangram_reference_import_v1')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['pairs','prior_conditions']},indent=2))
