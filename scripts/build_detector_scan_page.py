"""Prepare a private offline copy-and-record page; never call a detector."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(exported,workspace,round_number):
    manifest=json.loads((exported/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(exported/name)!=expected:raise ValueError('Frozen export changed')
    items=[json.loads(line) for line in (exported/'detector/PRIVATE_KEY.jsonl').read_text().splitlines()]
    selected=[r for r in items if r['round']==round_number]
    papers={r['paper_id'] for r in selected}
    if len(selected)!=20 or len(papers)!=4:raise ValueError('Complete four-paper round required')
    for paper in papers:
        group=[r for r in selected if r['paper_id']==paper]
        if len(group)!=5 or len({r['condition'] for r in group})!=5 or len({r['example_id'] for r in group})!=1:
            raise ValueError('Five matched conditions per case required')
    selected_ids={r['scan_id'] for r in selected}
    with (workspace/'results_working.csv').open() as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        worksheet=[r for r in reader if r['scan_id'] in selected_ids]
    if len(worksheet)!=20 or len({r['scan_id'] for r in worksheet})!=20:raise ValueError('Unique worksheet rows required')
    output=workspace/f'scan_round_{round_number}_pdf_only.html'
    if output.exists():raise ValueError('Do not overwrite a prepared scan page')
    by_id={r['scan_id']:r for r in selected};payload=[]
    for row in worksheet:
        item=by_id[row['scan_id']];text=(exported/'detector/texts'/f"{row['scan_id']}.txt").read_text()
        if hashlib.sha256(text.encode()).hexdigest()!=item['text_sha256']:raise ValueError('Exact text changed')
        payload.append({'record':row,'text':text})
    # Escape script delimiters; private paragraph contents never enter public source.
    data=json.dumps({'fields':fields,'items':payload},ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template='''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Private detector round __ROUND__</title>
<style>body{font:17px system-ui;max-width:1000px;margin:35px auto;padding:0 20px;color:#17212b}textarea{box-sizing:border-box;width:100%;height:280px;font:16px/1.6 Georgia}button,input,select{font:inherit;padding:7px;margin:5px 8px 8px 0}input{max-width:220px}label{display:inline-block}#message{min-height:1.5em;color:#245331}small{color:#52606d}hr{margin:24px 0}</style>
<h1>Private detector scans — round __ROUND__</h1>
<p>Twenty frozen texts, approximately __WORDS__ words. Check your remaining free allowance first.
Scan each text separately, unchanged, using the same provider snapshot. Include all twenty, even if a text has quality problems.
Do not combine them into one submission or regenerate them to improve scores.</p>
<p>This page works offline and makes no detector calls. It does not contain your credentials.
For each item, copy the text into GPTZero and download its basic-scan PDF.
Save the PDFs in the <b>pdf_reports</b> folder beside this page, or keep them in Downloads and tell Codex where they are.
Default filenames are fine: no renaming, CSV entry, labels, scores or dates need to be typed.</p>
<p>Codex will extract the first page, match the full scanned text to its frozen item, and record the results.
If a report omits information or does not match, it will be flagged rather than guessed.
If GPTZero cannot scan a text, skip it and mention that limitation when you return.</p>
<button id="previous">Previous</button><select id="position" aria-label="Scan item"></select><button id="next">Next</button>
<h2 id="title"></h2><small id="count"></small><p><button id="copy">Copy exact text</button><button id="downloadText">Download exact text</button></p>
<textarea id="text" readonly aria-label="Exact scan text"></textarea><div id="message" aria-live="polite"></div>
<script id="payload" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent),items=data.items;
let index=0;const byId=id=>document.getElementById(id);
const position=byId('position');items.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / 20 — '+x.record.scan_id;position.append(o)});
function show(){const x=items[index];position.value=index;byId('title').textContent=x.record.scan_id;byId('count').textContent=x.record.words_estimate+' whitespace-counted words';byId('text').value=x.text;byId('message').textContent=''}
position.onchange=()=>{index=Number(position.value);show()};byId('previous').onclick=()=>{index=Math.max(0,index-1);show()};byId('next').onclick=()=>{index=Math.min(items.length-1,index+1);show()};
byId('copy').onclick=async()=>{try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(items[index].text)}else{byId('text').focus();byId('text').select();if(!document.execCommand('copy'))throw Error('copy failed')}byId('message').textContent='Exact text copied.'}catch(e){byId('message').textContent='Select the text and copy it manually.'}};
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
byId('downloadText').onclick=()=>download(items[index].record.scan_id+'.txt',items[index].text,'text/plain;charset=utf-8');
show();
</script></html>'''
    content=template.replace('__ROUND__',str(round_number)).replace('__WORDS__',str(sum(len(x['text'].split()) for x in payload))).replace('__DATA__',data)
    output.write_text(content)
    receipt={'round':round_number,'texts':20,'words_estimate':sum(len(x['text'].split()) for x in payload),
        'export_manifest_sha256':digest(exported/'manifest.json'),'page_sha256':digest(output),'builder_sha256':digest(Path(__file__)),
        'status':'prepared_not_scanned','external_calls':0,'results_observed':0}
    (workspace/'pdf_reports').mkdir(exist_ok=True)
    (workspace/f'round_{round_number}_pdf_only_page_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Prepared private offline scan page: '+str(output)+'; no external calls.')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--exported',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--round',type=int,choices=(1,2),default=1)
    a=p.parse_args();build(a.exported,a.workspace,a.round)
