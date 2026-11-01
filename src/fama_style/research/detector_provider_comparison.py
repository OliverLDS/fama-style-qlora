"""Compare matched local GPTZero/Pangram reports, retaining native score semantics."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re

from .author_control_replication import verify_files
from .multi_author_detector import parse_gptzero_header
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def parse_pangram(text):
    label=re.search(r'We believe that this entire text is (AI-assisted|human-written|AI)\.',text)
    values=re.findall(r'(\d+(?:\.\d+)?)\s*%\s*of the document',text)
    versions=set(re.findall(r'Pangram (\d+(?:\.\d+)+)',text))
    day=re.search(r'\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b',text)
    count=re.search(r'\b(\d+) Words\s+Pangram',text)
    if not label or len(values)!=3 or len(versions)!=1:raise ValueError('unrecognized_pangram_header')
    values=list(map(float,values))
    if any(not 0<=x<=100 for x in values):raise ValueError('pangram_percentage_out_of_range')
    return {'provider_label':{'AI-assisted':'AI Assisted','human-written':'Human Written','AI':'AI Generated'}[label[1]],
        'provider_summary':label[0],'flagged':label[1]!='human-written','provider_version':next(iter(versions)),
        'report_date':datetime.strptime(day[1],'%b %d, %Y').date().isoformat() if day else None,
        'report_timezone':None,'ai_generated_document_percent':values[0],'ai_assisted_document_percent':values[1],
        'human_written_document_percent':values[2],'displayed_percentage_sum':sum(values),
        'percentages_rescaled':False,'score_interpretation':'Reported authorship shares, not document AI probability; tiny displayed shares may not sum to 100.',
        'provider_word_count':int(count[1]) if count else None,'numeric_decision_threshold':None,'pages_used_for_scores':[1]}


def layout_normalize(text):
    return re.sub(r'\s*([,.;:!?—–\-’‘“”"()\[\]])\s*',r'\1',' '.join(text.split()))


def cleaned_pages(pages):
    lines=[]
    for page in pages:
        for line in page.splitlines():
            if re.fullmatch(r'\s*(?:AI Assisted|AI|Human Written)\s+\d+\s+Words\s+(?:Low|Medium|High) Confidence\s*',line):continue
            if re.search(r'Page \d+/\d+\s*$',line):continue
            lines.append(line)
    return '\n'.join(lines)


def bind(report,items,pdf_sha,filename,confirmations):
    raw=' '.join(report['text'].split());layout=cleaned_pages(report['layout_pages'])
    matches=[r for r in items if ' '.join(r['text'].split()) in raw];method='raw_whitespace_exact'
    if not matches:
        matches=[r for r in items if layout_normalize(r['text']) in layout_normalize(layout)]
        method='visual_reading_order_punctuation_whitespace'
    if not matches:
        compact=lambda t:re.sub(r'\s+','',t)
        matches=[r for r in items if compact(r['text']) in compact(layout)]
        method='visually_confirmed_pdf_word_spacing_artifact'
        if len(matches)==1 and confirmations.get(pdf_sha)!={'scan_id':matches[0]['scan_id'],'text_sha256':matches[0]['text_sha256']}:
            raise ValueError('nonwhitespace_match_needs_versioned_visual_confirmation')
    if len(matches)!=1:raise ValueError('unique_complete_text_binding_required')
    item=matches[0];named=re.findall(r'SCAN_[0-9a-f]{16}',filename)
    if named and named!=[item['scan_id']]:raise ValueError('filename_conflicts_with_verified_text')
    return item,method


def compare(observations):
    by={(r['provider'],r['scan_id']):r for r in observations}
    if len(by)!=len(observations):raise ValueError('distinct_repeat_reports_need_separate_audit')
    ids={r['scan_id'] for r in observations if ('GPTZero',r['scan_id']) in by and ('Pangram',r['scan_id']) in by}
    cross=Counter();rows=[]
    for sid in sorted(ids):
        a,b=by[('GPTZero',sid)],by[('Pangram',sid)]
        cross[(a['provider_label'],b['provider_label'])]+=1
        rows.append({'scan_id':sid,'condition':a['condition'],'gptzero_label':a['provider_label'],'pangram_label':b['provider_label'],
            'ai_related_agreement':a['flagged']==b['flagged']})
    providers={}
    for provider in ('GPTZero','Pangram'):
        group=[r for r in observations if r['provider']==provider]
        conditions={}
        for condition in sorted({r['condition'] for r in group}):
            g=[r for r in group if r['condition']==condition]
            conditions[condition]={'n':len(g),'native_labels':dict(Counter(r['provider_label'] for r in g)),
                'ai_related_flags':sum(r['flagged'] for r in g)}
        shared=[r for r in group if r['scan_id'] in ids]
        human=[r for r in shared if r['condition']=='historical_human_reference']
        machine=[r for r in shared if r['condition']!='historical_human_reference']
        models=[r for r in shared if r['condition'] not in ('historical_human_reference','GPT6_input')]
        providers[provider]={'reports':len(group),'conditions':conditions,'matched_subset':{
            'human_references':len(human),'human_false_positive_flags':sum(r['flagged'] for r in human),
            'machine_produced_texts':len(machine),'machine_ai_related_flags':sum(r['flagged'] for r in machine),
            'gemma_outputs':len(models),'gemma_ai_related_flags':sum(r['flagged'] for r in models)}}
    return {'providers':providers,'shared_texts':len(ids),'ai_related_agreements':sum(r['ai_related_agreement'] for r in rows),
        'native_label_cross_tab':[{'gptzero':a,'pangram':b,'count':n} for (a,b),n in sorted(cross.items())]},rows


def account_batches(observations,provenance):
    entries=provenance['accounts'];mapping={r['pdf_sha256']:r['account_batch'] for r in entries}
    expected={r['pdf_sha256'] for r in observations if r['provider']=='Pangram'}
    if len(mapping)!=len(entries) or set(mapping)!=expected or any(not isinstance(v,str) or not v for v in mapping.values()):
        raise ValueError('account_provenance_must_cover_each_pangram_pdf_once')
    return dict(Counter(mapping.values()))


def build(exported,received,visual_audit,output,account_provenance=None):
    if output.exists():raise ValueError('refuse_overwrite_provider_comparison')
    verify_files(exported)
    items=[json.loads(l) for l in (exported/'detector/PRIVATE_KEY.jsonl').read_text().splitlines()]
    items=[r for r in items if r['round']==1]
    if len(items)!=20:raise ValueError('original_complete_round_required')
    audit=json.loads(visual_audit.read_text());confirmations={r['pdf_sha256']:{k:r[k] for k in ('scan_id','text_sha256')} for r in audit['confirmations']}
    for confirmation in audit['confirmations']:
        rendered=received/'visual_checks'/(confirmation['scan_id']+'.png')
        if file_sha(rendered)!=confirmation['rendered_first_page_sha256']:raise ValueError('visual_evidence_changed')
    observations=[];receipt={}
    for path in sorted((received/'raw_pdfs').glob('*.pdf')):
        extracted=received/'layout_extracted'/(path.name+'.private.json');report=json.loads(extracted.read_text())
        digest=file_sha(path);item,method=bind(report,items,digest,path.name,confirmations)
        first=report['text'];provider='Pangram' if 'Pangram' in first and 'AI Detection Report' in first else 'GPTZero'
        parsed=parse_pangram(first) if provider=='Pangram' else parse_gptzero_header(first)
        observations.append({k:item[k] for k in ('scan_id','example_id','paper_id','condition','text_sha256')}|
            {'provider':provider,'binding_method':method,'pdf_sha256':digest,'extracted_sha256':file_sha(extracted),**parsed})
        receipt[str(path.relative_to(received))]=digest;receipt[str(extracted.relative_to(received))]=file_sha(extracted)
    summary,paired=compare(observations)
    if summary['providers']['GPTZero']['reports']!=20:raise ValueError('complete_gptzero_round_required')
    observed={r['scan_id'] for r in observations if r['provider']=='Pangram'}
    pending=[{k:r[k] for k in ('scan_id','condition','example_id','paper_id','words_estimate')}|
        {'estimated_credits':math.ceil(r['words_estimate']/100)} for r in items if r['scan_id'] not in observed]
    summary.update(protocol='checkpoint_detector_comparison_v1',round=1,report_date_versions={p:{
        'dates':sorted({r['report_date'] for r in observations if r['provider']==p and r['report_date']}),
        'versions':sorted({r['provider_version'] for r in observations if r['provider']==p and r['provider_version']})} for p in ('GPTZero','Pangram')},
        score_semantics_comparable=False,all_scans_are_document_level_outcomes=True,sentence_level_analysis_required=False,
        pangram_pending=len(pending),pending_pangram_credits_estimate=sum(r['estimated_credits'] for r in pending),
        official_free_daily_credits=20,credit_estimate_uses_provider_word_count=False,
        statistically_equivalent_detectors_established=False,detector_calls_by_agent=0,training_called=False,
        binding_methods=dict(Counter(r['binding_method'] for r in observations)),
        matched_subset_selection='complete original four-paper round, five conditions per case' if summary['shared_texts']==20 else 'available matched overlap only; incomplete four-paper balance')
    if account_provenance is not None:
        provenance=json.loads(account_provenance.read_text());batches=account_batches(observations,provenance)
        summary.update(pangram_account_batches=batches,single_account_protocol_deviation=len(batches)>1,
            account_configuration_equivalence_verified=False,
            same_default_settings_user_reported=provenance.get('same_default_settings_user_reported'),
            account_provenance_basis='user reported; no account identities collected')
    output.mkdir(parents=True);write_rows(output/'observations.private.jsonl',observations);write_rows(output/'matched.private.jsonl',paired)
    if account_provenance is not None:write_json(output/'account_provenance.private.json',provenance)
    write_json(output/'pending_pangram.private.json',pending);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'export_manifest_sha256':file_sha(exported/'manifest.json'),
        'visual_audit_sha256':file_sha(visual_audit),'reports_and_extractions_sha256':receipt,
        **({'account_provenance_sha256':file_sha(account_provenance)} if account_provenance is not None else {})})
    write_json(output/'manifest.json',{'code_sha256':file_sha(Path(__file__)),
        'parser_sha256':file_sha(Path(__file__).with_name('multi_author_detector.py')),
        'extractor_sha256':file_sha(Path(__file__).resolve().parents[3]/'scripts/data/extract_detector_pdf_page.swift'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('exported','received','visual-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--account-provenance',type=Path)
    a=p.parse_args();print(json.dumps(build(a.exported,a.received,a.visual_audit,a.output,a.account_provenance)))


if __name__=='__main__':main()
