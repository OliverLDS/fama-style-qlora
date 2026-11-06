"""Import the six frozen mixed-500 detector PDFs with complete-text binding."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.research.detector_provider_comparison import bind
from fama_style.research.neutral_reconstruction_results import file_sha

SPEC=importlib.util.spec_from_file_location('pangram_parser',Path(__file__).with_name('import_corpus_200_pangram.py'))
parser=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(parser)


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def compare(observations):
    by={}
    for row in observations:
        key=(row['example_id'],row['condition'])
        if key in by:raise ValueError('repeat_report_requires_audit')
        by[key]=row
    ids=list(dict.fromkeys(r['example_id'] for r in observations))
    if len(ids)!=2 or len(by)!=6 or any((rid,c) not in by for rid in ids for c in ['input','base','epoch_1']):
        raise ValueError('two_complete_triplets_required')
    pairs=[]
    for rid in ids:
        source,base,trained=[by[rid,c] for c in ['input','base','epoch_1']]
        pairs.append(dict(example_id=rid,style_group=source['style_group'],paper_id=source['paper_id'],
            conditions={c:{k:by[rid,c][k] for k in ['provider_label','human_written_document_percent','ai_generated_document_percent','ai_assisted_document_percent','provider_word_count','material_output_issue_identified','source_word_sequence_overlap']} for c in ['input','base','epoch_1']},
            change_human_share_vs_input=trained['human_written_document_percent']-source['human_written_document_percent'],
            change_human_share_vs_base=trained['human_written_document_percent']-base['human_written_document_percent'],
            AI_related_input_becomes_entirely_human=not source['entirely_human'] and trained['entirely_human']))
    pairs.sort(key=lambda r:r['style_group'])
    return dict(cases=2,reports=6,pairs=pairs,
        entirely_human_counts={c:sum(by[rid,c]['entirely_human'] for rid in ids) for c in ['input','base','epoch_1']},
        higher_human_share_than_input=sum(r['change_human_share_vs_input']>0 for r in pairs),
        lower_human_share_than_input=sum(r['change_human_share_vs_input']<0 for r in pairs),
        detector_calls_by_agent=0,new_model_calls=0,new_GPU_jobs=0,
        score_semantics='Displayed authorship shares, not document AI probabilities. Values and display floors retained without rescaling.',
        limits=['Two preselected contents, not a representative benchmark or a general success estimate.',
            'Native AI Generated and AI Assisted outcomes remain distinct.',
            'Inflation output contains literal HTML subscript formatting; a detector confound.',
            'A human-written label is not evidence of Fama-style learning or independently verified meaning.',
            'Semantic assessments are provisional same-agent judgments; settings/account equivalence not independently verified.'])


def build(run,output):
    if output.exists():raise ValueError('refuse_overwrite_detector_import')
    packet=run/'review_v1/pangram_scans_v1';received=run/'pangram_reports_v1'
    for name,h in read(packet/'manifest.json')['files'].items():
        if file_sha(packet/name)!=h:raise ValueError('frozen_scan_packet_changed')
    review=run/'review_v1'
    for name,h in read(review/'manifest.json')['files'].items():
        if file_sha(review/name)!=h:raise ValueError('review_changed')
    items=rows(packet/'PRIVATE_KEY.jsonl')
    if len(items)!=6:raise ValueError('six_scan_texts_required')
    outputs={(r['example_id'],r['snapshot']):r for r in rows(review/'outputs_and_assessments.private.jsonl')}
    observed=set();reports=[];evidence={}
    for e in read(received/'receipt.private.json'):
        h=e['pdf_sha256'];pdf=received/'raw_pdfs'/(h+'.pdf');ex=received/'extracted'/(h+'.json')
        if file_sha(pdf)!=h or file_sha(ex)!=e['extracted_sha256']:raise ValueError('PDF_evidence_changed')
        page=read(ex);item,method=bind(page,items,h,Path(e['source']).name,{})
        if item['scan_id'] in observed:raise ValueError('repeat_report_requires_audit')
        if file_sha(packet/'texts'/(item['scan_id']+'.txt'))!=item['text_sha256']:raise ValueError('exact_scan_text_changed')
        observed.add(item['scan_id'])
        original=outputs[item['example_id'],'base' if item['condition']=='input' else item['condition']]
        if item['condition']!='input' and original['output_sha256']!=item['text_sha256']:raise ValueError('output_binding')
        if item['condition']=='input' and __import__('hashlib').sha256(original['source'].encode()).hexdigest()!=item['text_sha256']:raise ValueError('input_binding')
        reports.append(dict(**{k:item[k] for k in ['scan_id','example_id','condition','text_sha256']},paper_id=original['paper_id'],
            style_group=original['style_group'],pdf_sha256=h,binding_method=method,
            material_output_issue_identified=None if item['condition']=='input' else original['assessment']['material_claim_change'],
            source_word_sequence_overlap=1.0 if item['condition']=='input' else original['source_metrics']['word_sequence_overlap'],
            **parser.parse(page['text'])))
        evidence[str(pdf.relative_to(received))]=h;evidence[str(ex.relative_to(received))]=e['extracted_sha256']
    if observed!={r['scan_id'] for r in items}:raise ValueError('incomplete_scan_coverage')
    summary=compare(reports)
    summary.update(protocol='corpus_500_pangram_v1',provider_versions=sorted({r['provider_version'] for r in reports}),
        report_dates=sorted({r['report_date'] for r in reports}),binding_methods=dict(Counter(r['binding_method'] for r in reports)))
    output.mkdir();(output/'observations.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in reports))
    write(output/'summary.json',summary)
    write(output/'manifest.json',dict(builder_sha256=file_sha(Path(__file__)),parser_sha256=file_sha(Path(parser.__file__)),
        binding_sha256=file_sha(ROOT/'src/fama_style/research/detector_provider_comparison.py'),
        scan_manifest_sha256=file_sha(packet/'manifest.json'),review_manifest_sha256=file_sha(review/'manifest.json'),
        receipt_sha256=file_sha(received/'receipt.private.json'),evidence=evidence,
        files={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');a=p.parse_args()
    print(json.dumps(build(a.run,a.run/'pangram_import_v1'),indent=2))
