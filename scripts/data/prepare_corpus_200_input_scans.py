"""Prepare the exact eight missing input controls for the completed output scans."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.evaluation import corpus_200_simple_prompt as runtime


def read(path): return json.loads(path.read_text())
def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def select(original, simple):
    ids = original['config']['detector_case_ids']
    if len(ids) != 8 or len(set(ids)) != 8 or simple['previous_detector_case_ids'] != ids:
        raise ValueError('same_eight_prefrozen_cases_required')
    maps = []
    for report in [original, simple]:
        for snap in report['snapshots']:
            rows = snap['generations']
            if len({r['example_id'] for r in rows}) != len(rows): raise ValueError('duplicate_input_id')
            maps.append({r['example_id']: r for r in rows})
    if len(maps) != 4: raise ValueError('four_output_conditions_required')
    result = []
    for rid in ids:
        row = maps[0][rid]
        for mapping in maps[1:]:
            if any(mapping[rid][k] != row[k] for k in ['source', 'paper_id', 'style_group']):
                raise ValueError('input_changed_between_conditions')
        text = row['source']; count = len(text.split())
        result.append(dict(scan_id='SCAN_' + runtime.sha('corpus200-exact-input-v1|' + rid)[:16],
                           example_id=rid, paper_id=row['paper_id'], style_group=row['style_group'],
                           condition='actual_inference_input', text=text, text_sha256=runtime.sha(text),
                           words_estimate=count, credits_estimate=math.ceil(count / 100),
                           provider_minimum_met=count >= 50))
    return result


def build(parent, comparison, output):
    if output.exists(): raise ValueError('refuse_overwrite_input_controls')
    prep = read(comparison / 'preparation.json')
    path = parent / 'raw/corpus_200_pilot_v1/evaluation.json'
    _, original = runtime.load_inputs(path, path.parent / 'epoch_1/adapter', prep['parent_evaluation_sha256'])
    simple_path = comparison / 'raw/corpus_200_simple_prompt_v1/evaluation.json'
    receipt = read(comparison / 'download_manifest.json')
    if file_sha(simple_path) != receipt['files']['corpus_200_simple_prompt_v1/evaluation.json']:
        raise ValueError('comparison_report_changed')
    items = select(original, read(simple_path))
    credits = sum(r['credits_estimate'] for r in items if r['provider_minimum_met'])
    if credits > 20: raise ValueError('packet_budget_exceeded_do_not_truncate')
    audit = []; searched = []; observations = 0
    for p in sorted((ROOT / 'research_runs').rglob('*observations*.jsonl')):
        if output in p.parents: continue
        searched.append(dict(path=str(p.relative_to(ROOT)), sha256=file_sha(p)))
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            record = json.loads(line); observations += 1
            for item in items:
                if record.get('text_sha256') == item['text_sha256']:
                    audit.append(dict(example_id=item['example_id'], artifact=str(p.relative_to(ROOT)),
                                      provider=record.get('provider'), provider_label=record.get('provider_label'),
                                      pdf_sha256=record.get('pdf_sha256')))
    output.mkdir(); (output / 'texts').mkdir(); (output / 'pdf_reports').mkdir()
    for r in items: (output / 'texts' / (r['scan_id'] + '.txt')).write_text(r['text'])
    (output / 'PRIVATE_KEY.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    write(output / 'prior_reports_audit.private.json', dict(files=searched, observation_records=observations,
          exact_text_hash_matches=audit, scope='Imported observation records only; no claim about unimported external scans'))
    cards = [{k: r[k] for k in ['scan_id', 'text', 'words_estimate', 'credits_estimate', 'provider_minimum_met']} for r in items]
    data = json.dumps(cards, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Eight actual inference inputs</title>
<style>body{max-width:900px;margin:30px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:330px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}</style>
<h1>Eight actual inference inputs</h1><p>These are the unchanged AI-generated inputs to the 32 outputs already scanned.
Scan each paragraph separately with the same Pangram settings. Copy only the text, with no scan ID or title.
Do not edit, shorten or combine it. Estimated total: __CREDITS__ credits; actual word counts and balance may differ.</p>
<p>Save every PDF into the <b>pdf_reports</b> folder beside this page. Default filenames are fine. No CSV or manual score entry.
Keep all outcomes, including human-written labels. If a scan fails, retain the unchanged text and report the missing item.</p>
<button id="prev">Previous</button><select id="pick" aria-label="Item"></select><button id="next">Next</button>
<h2 id="name"></h2><p id="cost"></p><button id="copy">Copy exact text</button><textarea id="text" readonly></textarea><p id="msg"></p>
<script type="application/json" id="data">__DATA__</script><script>
const items=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let at=0;
items.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+items.length+' — '+x.scan_id;el('pick').append(o)});
function show(){let x=items[at];el('pick').value=at;el('name').textContent=x.scan_id;el('text').value=x.text;el('cost').textContent=x.words_estimate+' estimated words; '+x.credits_estimate+' estimated credits';el('msg').textContent=x.provider_minimum_met?'':'Below minimum; leave unscanned, do not pad.'}
el('pick').onchange=()=>{at=Number(el('pick').value);show()};el('prev').onclick=()=>{at=Math.max(0,at-1);show()};el('next').onclick=()=>{at=Math.min(items.length-1,at+1);show()};
el('copy').onclick=async()=>{try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(items[at].text);else{el('text').focus();el('text').select();if(!document.execCommand('copy'))throw Error('copy')}el('msg').textContent='Exact input copied.'}catch(e){el('msg').textContent='Select the text and copy it manually.'}};show();
</script></html>'''
    (output / 'scan.html').write_text(page.replace('__CREDITS__', str(credits)).replace('__DATA__', data))
    write(output / 'summary.json', dict(protocol='corpus_200_exact_input_controls_v1', status='prepared_not_scanned',
          texts=8, estimated_credits=credits, prior_observation_files=len(searched),
          prior_observation_records=observations, exact_prior_report_matches=len(audit),
          output_rescans_required=0, reference_scans_requested=0, new_model_calls=0, new_detector_calls=0,
          selection='Exact same eight cases as the already scanned output panel; no score-based reselection'))
    write(output / 'manifest.json', dict(builder_sha256=file_sha(Path(__file__)),
          parent_report_sha256=file_sha(path), simple_report_sha256=file_sha(simple_path),
          files={str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return read(output / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'comparison', 'output']: parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(build(a.parent, a.comparison, a.output), indent=2))
