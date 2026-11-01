"""Import GPTZero PDFs without user-entered scores, CSV fields or filenames.

Exact scanned-text matching binds reports to frozen items. No detector calls.
Unknown layouts, truncated text and conflicting repeat scans remain explicit.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import subprocess

from .author_control_replication import verify_files
from .multi_author_detector import parse_first_page
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def match_page(text,items):
    normalized=' '.join(text.split())
    return [r for r in items if ' '.join(r['text'].split()) in normalized]


def bind_page(page,items):
    if page.get('page_extracted')!=1:raise ValueError('page_one_required')
    matched=match_page(page.get('text',''),items)
    if len(matched)!=1:raise ValueError('ambiguous_text_match' if matched else 'no_complete_exact_text_match')
    item=matched[0]
    return {'scan_id':item['scan_id'],'text_sha256':item['text_sha256'],**parse_first_page(page['text'],item['text'])}


def extract_first_page(pdf,target):
    if shutil.which('swift'):
        script=Path(__file__).resolve().parents[3]/'scripts/data/extract_detector_pdf_page.swift'
        subprocess.run(['swift','-module-cache-path','/tmp/fama-detector-pdf-cache',str(script),str(pdf),str(target)],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:
        try:from pypdf import PdfReader
        except ImportError:raise RuntimeError('Local PDF extraction needs macOS Swift/PDFKit or pypdf') from None
        reader=PdfReader(str(pdf));write_json(target,{'page_count':len(reader.pages),'page_extracted':1,'text':reader.pages[0].extract_text() or ''})
    return json.loads(target.read_text())


def results_table(items,observations):
    by={}
    for r in observations:
        if r['status']=='parsed':by.setdefault(r['scan_id'],[]).append(r)
    rows=[]
    for item in items:
        matches=by.get(item['scan_id'],[])
        row={k:item[k] for k in ('scan_id','text_sha256','round')}
        row.update(provider='GPTZero',status='not_tested',provider_label='',score='',score_scale='',test_date='',
            provider_version='',threshold_rule='',notes='')
        if len(matches)>1:
            row.update(status='conflicting_repeat_reports',notes='Multiple distinct PDFs bind to this text; no preferred score selected.')
        elif matches:
            report=matches[0];row.update(status='scored',provider_label=report['provider_label'],score=report['ai_probability_percent'],
                score_scale='0-100 document AI probability',test_date=report['report_date'] or '',
                provider_version=report['provider_version'] or 'not shown',threshold_rule='provider-native label; numeric threshold not shown',
                notes='Page 1; report date has unknown timezone. PDF SHA256 '+report['pdf_sha256'])
        rows.append(row)
    return rows


def build(exported,pdfs,output,round_number=1):
    if output.exists():raise ValueError('refuse_overwrite_pdf_import')
    verify_files(exported)
    items=[json.loads(line) for line in (exported/'detector/PRIVATE_KEY.jsonl').read_text().splitlines()]
    items=[r for r in items if r['round']==round_number]
    if len(items)!=20 or len({r['scan_id'] for r in items})!=20:raise ValueError('frozen_twenty_text_round_required')
    paths=sorted(p for p in pdfs.rglob('*') if p.is_file() and p.suffix.lower()=='.pdf')
    if not paths:raise ValueError('no_pdf_reports_found')
    (output/'raw_pdfs').mkdir(parents=True);(output/'extracted').mkdir()
    observations=[];seen=set();aliases=[]
    for path in paths:
        digest=file_sha(path)
        if digest in seen:
            aliases.append({'source_filename':str(path.relative_to(pdfs)),'pdf_sha256':digest});continue
        seen.add(digest);local=output/'raw_pdfs'/f'{digest}.pdf';shutil.copyfile(path,local)
        target=output/'extracted'/f'{digest}.private.json'
        entry={'source_filename':str(path.relative_to(pdfs)),'pdf_sha256':digest,'status':'unresolved'}
        try:
            page=extract_first_page(local,target)
            entry.update(bind_page(page,items),status='parsed',extracted_page_sha256=file_sha(target))
        except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as error:
            entry['reason']=str(error) if isinstance(error,ValueError) else type(error).__name__
            if target.exists():entry['extracted_page_sha256']=file_sha(target)
        observations.append(entry)
    table=results_table(items,observations)
    write_rows(output/'observations.private.jsonl',observations);write_json(output/'duplicate_file_aliases.private.json',aliases)
    with (output/'results_imported.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    statuses=Counter(r['status'] for r in table)
    summary={'protocol':'checkpoint_gptzero_pdf_import_v1','round':round_number,'expected_texts':20,'pdf_files_received':len(paths),
        'unique_pdf_files':len(seen),'identical_duplicate_files':len(aliases),'scored_texts':statuses['scored'],
        'missing_texts':statuses['not_tested'],'conflicting_texts':statuses['conflicting_repeat_reports'],
        'unresolved_reports':sum(r['status']=='unresolved' for r in observations),'complete_round':statuses['scored']==20,
        'external_calls':0,'manual_csv_entry_required':False,'user_renaming_required':False,
        'missing_metadata_fabricated':False,'report_matching':'full scanned text, whitespace normalization only',
        'provider_versions_observed':sorted({r['provider_version'] for r in table if r['status']=='scored'}),
        'report_dates_observed':sorted({r['test_date'] for r in table if r['status']=='scored'})}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'code_sha256':file_sha(Path(__file__)),
        'export_manifest_sha256':file_sha(exported/'manifest.json'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('exported','pdfs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--round',type=int,choices=(1,2),default=1);a=p.parse_args()
    print(json.dumps(build(a.exported,a.pdfs,a.output,a.round)))


if __name__=='__main__':main()
