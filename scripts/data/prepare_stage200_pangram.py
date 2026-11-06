"""Prepare eight blinded dataset-control texts; no detector or model calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build(cohort, output):
    if output.exists():
        raise ValueError('Refuse to overwrite a frozen scan packet')
    manifest=json.loads((cohort.parent/'manifest.json').read_text())
    if hashlib.sha256(cohort.read_bytes()).hexdigest()!=manifest['outputs'][cohort.name]:
        raise ValueError('Frozen cohort hash changed')
    rows=[json.loads(l) for l in cohort.read_text().splitlines()]
    if len(rows)!=200: raise ValueError('Expected frozen 200-pair cohort')
    audit_path=Path('research_runs/author_pair_expansion_v1/response_audit_v1/audits.private.jsonl')
    originals=[json.loads(l) for l in audit_path.read_text().splitlines()]
    groups=sorted({r['style_group'] for r in rows})
    selected=[];eligible_counts={}
    for group in groups:
        eligible=[]
        for r in rows:
            if r['style_group']!=group: continue
            p=r['provenance']
            # This diagnostic compares known GPT-6 neutral rewrites against
            # human targets. Exclude new Codex inputs and unestablished legacy
            # model origins; retain the cohort unchanged.
            if 'parent_index' not in p: continue
            a=originals[p['parent_index']]
            if a['paragraph_id']!=r['parent_id']: raise ValueError('Parent mismatch')
            if r['plain_english']!=a['annotation']['plain_english']: continue
            lengths=[len(r[k].split()) for k in ('plain_english','original_paragraph')]
            if all(100<=n<=200 for n in lengths): eligible.append(r)
        eligible_counts[group]=len(eligible)
        if not eligible: raise ValueError('No full-text budget-eligible pair: '+group)
        selected.append(min(eligible,key=lambda r:sha('stage200-pangram-controls-v1|'+r['id'])))
    items=[]
    for row in selected:
        for field,condition in [('plain_english','GPT6_neutral_input'),('original_paragraph','historical_human_target')]:
            text=row[field]
            items.append({'scan_id':'SCAN_'+sha('stage200|'+row['id']+'|'+field)[:16],
                          'example_id':row['id'],'parent_id':row['parent_id'],'paper_id':row['paper_id'],
                          'style_group':row['style_group'],'condition':condition,'text':text,
                          'text_sha256':sha(text),'words_estimate':len(text.split()),
                          'credits_estimate':math.ceil(len(text.split())/100),
                          'source_provenance':row['provenance'],
                          'human_target_transcription_status':row['quality']})
    items.sort(key=lambda r:sha('stage200-pangram-blind-order-v1|'+r['scan_id']))
    credits=sum(r['credits_estimate'] for r in items)
    assert len(items)==8 and credits<=20
    (output/'texts').mkdir(parents=True);(output/'pdf_reports').mkdir()
    for r in items:(output/'texts'/f"{r['scan_id']}.txt").write_text(r['text'])
    (output/'PRIVATE_KEY.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    cards=[{k:r[k] for k in ['scan_id','text','words_estimate','credits_estimate']} for r in items]
    payload=json.dumps(cards,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Stage 200 — Pangram dataset controls</title>
<style>body{max-width:900px;margin:35px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:330px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}small{color:#465465}</style>
<h1>Pangram dataset controls — eight scans</h1>
<p>This packet checks training-data detectability. It contains no fine-tuned Gemma output.
Scan the full text for each item separately, unchanged, with the same default AI-detection settings.
Do not include the scan ID in the scanned text or enable a plagiarism check.</p>
<p>Estimated total: __CREDITS__ credits. Check your remaining balance first.
Download each Pangram report as a PDF into <b>pdf_reports</b> beside this page.
Default filenames are fine. No CSV fields, scores or manual renaming are required.
Keep every report, including unexpected results. If scanning is unavailable, leave the item unscanned.</p>
<p><a href="https://www.pangram.com/">Open Pangram</a></p>
<button id="prev">Previous</button><select id="pick" aria-label="Scan item"></select><button id="next">Next</button>
<h2 id="name"></h2><small id="cost"></small><p><button id="copy">Copy exact text</button></p>
<textarea id="text" readonly aria-label="Exact scan text"></textarea><p id="msg" aria-live="polite"></p>
<script type="application/json" id="data">__DATA__</script><script>
const items=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let at=0;
items.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+items.length+' — '+x.scan_id;el('pick').append(o)});
function show(){const x=items[at];el('pick').value=at;el('name').textContent=x.scan_id;el('text').value=x.text;el('cost').textContent=x.words_estimate+' estimated words; '+x.credits_estimate+' credits';el('msg').textContent=''}
el('pick').onchange=()=>{at=Number(el('pick').value);show()};el('prev').onclick=()=>{at=Math.max(0,at-1);show()};el('next').onclick=()=>{at=Math.min(items.length-1,at+1);show()};
el('copy').onclick=async()=>{try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(items[at].text)}else{el('text').focus();el('text').select();if(!document.execCommand('copy'))throw Error('copy')}el('msg').textContent='Exact text copied.'}catch(e){el('msg').textContent='Select the text and copy it manually.'}};show();
</script></html>'''.replace('__DATA__',payload).replace('__CREDITS__',str(credits))
    (output/'scan.html').write_text(page)
    summary={'protocol':'stage200_pangram_dataset_controls_v1','status':'prepared_not_scanned',
             'purpose':'Dataset-side detector separation; no inference or fine-tuning effectiveness claim',
             'cohort_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest(),
             'pairs':4,'texts':8,'estimated_credits':credits,'eligible_pairs_per_group':eligible_counts,
             'selection':'Fixed hash, one pair per group; unchanged known GPT-6 input; both full texts 100–200 whitespace-counted words',
             'selection_limit':'Length/provenance-restricted purposive diagnostic, not a representative estimate for all 200 pairs',
             'all_cohort_texts_estimated_credits':sum(math.ceil(len(r[k].split())/100) for r in rows for k in ['plain_english','original_paragraph']),
             'additional_model_calls':0,'GPU_jobs':0,'detector_calls':0,'observed_results':0,
             'policy_source':'https://www.pangram.com/knowledge-hub/what-is-an-ai-detection-credit',
             'policy_checked':'2026-11-03','policy':'20 free credits/day; one per 100 words rounded up; actual provider counting/balance may differ',
             'next':'Import PDF reports by matching complete text and retain native labels, authorship shares, date and version; do not filter the training cohort by scores.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'START_HERE.md').write_text('# Pangram dataset controls\n\nOpen `scan.html`. Scan its eight texts separately and save the eight PDF reports to `pdf_reports/`. Default names are fine; no CSV entry is needed. Estimated total: '+str(credits)+' credits.\n\nThis checks the dataset inputs and targets, not a trained model. No detector calls have been made. The frozen training cohort is unchanged.\n')
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps({'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cohort_sha256':summary['cohort_sha256'],'parent_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'files':files},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.cohort,a.output)
