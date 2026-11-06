"""Prepare sparse Pangram comparisons for both Gemini input routes.

Reuses exact prior input scans and published-human calibration. No detector,
provider or GPU calls. Native texts remain unchanged, including format failures.
"""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('purpose_prompt_review',Path(__file__).with_name('review_purpose_500_prompt_input.py'))
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
RUN=review.DOWNLOAD
OUTPUT=RUN/'review_v1/pangram_scans_v1'
HISTORY=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1'


def verify_files(folder):
    manifest=review.read(folder/'manifest.json')
    for name,h in manifest['files'].items():
        if review.runtime.file_sha(folder/name)!=h:raise ValueError('frozen_evidence_changed')
    return manifest


def prior_observations():
    full=HISTORY/'pangram_full_import_v1';manifest=verify_files(full)
    previous=HISTORY/'pangram_import_v1';initial=verify_files(previous)
    if manifest['initial_import_manifest_sha256']!=review.runtime.file_sha(previous/'manifest.json'):raise ValueError('prior_import_changed')
    for record,received in [(manifest,HISTORY/'pangram_remaining_reports_v1'),(initial,HISTORY/'pangram_reports_v1')]:
        for name,h in record['evidence'].items():
            if review.runtime.file_sha(received/name)!=h:raise ValueError('prior_PDF_evidence_changed')
    return review.rows(full/'observations.private.jsonl')


def select(items,previous):
    previous_by_hash={}
    for r in previous:previous_by_hash.setdefault(r['text_sha256'],[]).append(r)
    chosen=[r for r in items if r['input_condition'] in ['gemini_input','fresh_reconstruction'] and r['prompt_condition']=='reexpress']
    if len(chosen)!=16:raise ValueError('two_routes_eight_matched_cases_required')
    by={(r['content_id'],r['input_condition'],r['snapshot']):r for r in chosen}
    cards=[]
    for cid,route in sorted({(r['content_id'],r['input_condition']) for r in chosen}):
        base=by[cid,route,'base'];trained=by[cid,route,'epoch_1']
        if any(base[k]!=trained[k] for k in ['source','reference','source_sha256','paper_id','style_group','prompt_condition','prompt_sha256']):raise ValueError('matched_input_changed')
        for condition,text in [('input',base['source']),('base',base['output']),('epoch_1',trained['output'])]:
            h=review.runtime.sha(text);n=len(text.split())
            matches=previous_by_hash.get(h,[])
            # Reuse an exact observation, with its native label/date/version;
            # conflicting repeat scans require a new audit, never pick a score.
            if len({(r['provider_label'],r['human_written_document_percent'],r['ai_generated_document_percent'],r['ai_assisted_document_percent']) for r in matches})>1:
                raise ValueError('conflicting_historical_scores_require_audit')
            observation=matches[0] if matches else None
            failed_task=(route=='fresh_reconstruction' and condition=='epoch_1' and
                         base['style_group']=='Shiller' and n<50)
            status='reused_exact_prior_observation' if observation else 'not_scanned_task_failure' if failed_task else 'new_scan_required'
            if n<50 and not failed_task:raise ValueError('short_text_requires_explicit_audit_no_padding')
            cards.append(dict(scan_id='SCAN_'+review.runtime.sha('purpose500-reexpress-pangram-v1|'+cid+'|'+route+'|'+condition)[:16],
                content_id=cid,input_condition=route,prompt_condition='reexpress',condition=condition,
                paper_id=base['paper_id'],style_group=base['style_group'],text=text,text_sha256=h,
                words_estimate=n,credits_estimate=math.ceil(n/100) if status=='new_scan_required' else 0,
                status=status,round=1 if route=='gemini_input' else 2,
                prior_observation=observation,
                failure_note='Known 15-word missing-passage request; retain as task failure, not a humanization success.' if failed_task else None,
                content_review_note=None if condition=='input' else (base if condition=='base' else trained).get('reading_note')))
    if len(cards)!=24 or len({r['scan_id'] for r in cards})!=24:raise ValueError('panel_coverage')
    return cards


def render(cards,round_number,cost):
    # This page has only texts requiring a new scan; reused and failed entries
    # stay in the complete private key and coverage report.
    data=json.dumps([{k:r[k] for k in ['scan_id','text','words_estimate','credits_estimate']} for r in cards],ensure_ascii=False)
    data=data.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pangram — saved 500-pair adapter</title>
<style>body{max-width:900px;margin:30px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:400px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}p{line-height:1.5}</style>
<h1>Pangram — round __ROUND__</h1><p>__COUNT__ new scans, approximately __COST__ credits using your earlier accounting rule. Actual provider counts may differ.</p>
<p>Use your usual default settings. Copy each full text exactly, without its ID. Keep headings, prefatory text and mathematical notation as shown. Do not shorten, edit or combine texts.</p>
<p>Export each report as a PDF and save it in <b>pdf_reports</b> beside this page. Default filenames are fine; no CSV, renaming or manual score entry is needed. Preserve every result. If credits run out, continue after the normal quota refresh.</p>
<button id="prev">Previous</button><select id="pick" aria-label="Scan text"></select><button id="next">Next</button><h2 id="name"></h2><p id="cost"></p><button id="copy">Copy exact text</button><p id="message"></p><textarea id="text" readonly aria-label="Full scan text"></textarea>
<script>const cards=__DATA__;const pick=document.getElementById('pick');
cards.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+cards.length+' — '+c.scan_id;pick.append(o)});
function show(){const c=cards[+pick.value];document.getElementById('name').textContent=c.scan_id;document.getElementById('text').value=c.text;document.getElementById('cost').textContent=c.words_estimate+' words; estimated '+c.credits_estimate+' credits';document.getElementById('message').textContent=''}
pick.onchange=show;document.getElementById('prev').onclick=()=>{pick.value=Math.max(0,+pick.value-1);show()};document.getElementById('next').onclick=()=>{pick.value=Math.min(cards.length-1,+pick.value+1);show()};
document.getElementById('copy').onclick=async()=>{let t=document.getElementById('text');try{await navigator.clipboard.writeText(t.value)}catch(e){t.select();document.execCommand('copy')}document.getElementById('message').textContent='Copied exact text.'};show();</script></html>'''.replace('__ROUND__',str(round_number)).replace('__COUNT__',str(len(cards))).replace('__COST__',str(cost)).replace('__DATA__',data)


def build():
    if OUTPUT.exists():raise ValueError('refuse_overwrite_Pangram_packet')
    verify_files(RUN/'review_v1')
    review.verify()
    items=review.rows(RUN/'review_v1/outputs_and_reading_notes.private.jsonl')
    previous=prior_observations();cards=select(items,previous)
    OUTPUT.mkdir();(OUTPUT/'texts').mkdir();(OUTPUT/'pdf_reports').mkdir()
    write_rows=lambda p,r:p.write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in r))
    write_rows(OUTPUT/'PRIVATE_KEY.jsonl',cards)
    write_rows(OUTPUT/'reused_observations.private.jsonl',[r for r in cards if r['status']=='reused_exact_prior_observation'])
    write_rows(OUTPUT/'unscannable_failures.private.jsonl',[r for r in cards if r['status']=='not_scanned_task_failure'])
    rounds=[]
    for number in [1,2]:
        new=[r for r in cards if r['status']=='new_scan_required' and r['round']==number]
        new.sort(key=lambda r:review.runtime.sha('purpose500-pangram-order|'+r['scan_id']))
        cost=sum(r['credits_estimate'] for r in new)
        if cost>20:raise ValueError('round_exceeds_earlier_daily_budget_do_not_truncate')
        for r in new:(OUTPUT/'texts'/(r['scan_id']+'.txt')).write_text(r['text'])
        (OUTPUT/f'scan_round_{number}.html').write_text(render(new,number,cost))
        rounds.append(dict(round=number,new_scans=len(new),estimated_credits=cost))
    summary=dict(protocol='purpose_500_reexpress_pangram_v1',status='prepared_not_scanned',
        contents=4,input_cases=8,input_routes=['gemini_input','fresh_reconstruction'],prompt_condition='reexpress',
        full_panel_text_slots=24,coverage=dict(Counter(r['status'] for r in cards)),rounds=rounds,
        new_scans=sum(r['status']=='new_scan_required' for r in cards),estimated_credits=sum(r['credits_estimate'] for r in cards),
        human_reference_rescans=0,prior_human_calibration='Previously published human paragraphs: 8/8 Human Written at 100% plus earlier genre controls; no routine repeat needed.',
        decision_value='Compare adapted output with its actual Gemini input and base output under the revised prompt, including fresh reconstructions; retain failure and ceiling cases.',
        input_ceiling_cases=sum(r['condition']=='input' and r['prior_observation'] is not None and r['prior_observation']['entirely_human'] for r in cards),
        no_scan_on_failed_short_response=True,failure_counts_as_success=False,
        score_semantics='Native authorship shares and labels; not document AI probabilities.',
        limits=['Four contents reused across two Gemini input routes; paired exploratory measurements, not independent cases.',
            'Reused prior scans retain their dates and Pangram version; assess version differences when new PDFs arrive.',
            'The known task failure remains a failed case with unavailable detector score, not an excluded success denominator.',
            'Headings and prefatory text are preserved and may affect detector results.',
            'Detector reduction does not alone establish readability, meaning preservation or Fama-specific style.'],
        new_detector_calls=0,new_model_calls=0,new_GPU_jobs=0)
    review.runtime.write_json(OUTPUT/'summary.json',summary)
    links=''.join('<p><a href="scan_round_%d.html">Round %d: %d scans, approximately %d credits</a></p>'%(r['round'],r['round'],r['new_scans'],r['estimated_credits']) for r in rounds)
    (OUTPUT/'start_here.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Gemini-input Pangram comparison</title><body style="font:18px system-ui;max-width:850px;margin:40px auto;padding:20px"><h1>Gemini inputs → base and adapted rewrites</h1><p>19 new scans, approximately 29 credits. Four exact prior input scans are reused. No published-human rescans. One short missing-passage response remains a failed rewrite and is not submitted.</p>'+links+'<p>Use normal daily quota refreshes as needed. Save all 19 exported PDFs, using their default filenames, in the <b>pdf_reports</b> folder beside this page. No CSV or manual score entry.</p><p>The revised re-expression prompt is fixed. Original Gemini inputs were not chosen for favorable detector scores, and one was already labeled human. Report input/output changes separately from adapter/base changes.</p></body></html>')
    (OUTPUT/'START_HERE.md').write_text('Open start_here.html. Round 1: 8 scans, estimated 13 credits. Round 2: 11 scans, estimated 16 credits. Save all 19 default-named Pangram PDF exports in pdf_reports/. No manual score entry. Use normal quota refreshes. Four old input scans are reused; no human references are requested. The known short refusal stays a task failure.\n')
    review.runtime.write_json(OUTPUT/'manifest.json',dict(builder_sha256=review.runtime.file_sha(Path(__file__)),
        review_manifest_sha256=review.runtime.file_sha(RUN/'review_v1/manifest.json'),
        prior_full_import_manifest_sha256=review.runtime.file_sha(HISTORY/'pangram_full_import_v1/manifest.json'),
        human_calibration_doc_sha256=review.runtime.file_sha(ROOT/'docs/data_scaling_v1/STAGE_0500_REFERENCE_CONTROLS_V1.md'),
        files={str(p.relative_to(OUTPUT)):review.runtime.file_sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':print(json.dumps(build(),indent=2))
