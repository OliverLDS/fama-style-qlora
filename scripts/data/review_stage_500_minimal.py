"""Meaning-only patch review that cannot silently replace neutral prose with human targets."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

SPEC=importlib.util.spec_from_file_location('expansion',Path(__file__).with_name('expand_stage_500.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
RUN=m.RUN/'minimal_review_v1'
PROMPT='''Compare each HUMAN paragraph and its NEUTRAL rewrite for ECONOMIC MEANING ONLY.
Do NOT restore the author's wording or style. Synonyms, reordered sentences, spelling variants, clear paraphrases,
and modest tone differences are acceptable. Do not change neutral prose merely to make it resemble HUMAN.
Minor softening of evaluative language is not a reason to rewrite. Footnote-anchor markers without footnote text are typography, not economic quantities.
Check missing substantive propositions, definitions, actors, conditions, causal direction, negation, actual quantities and comparisons.
Be especially careful with the supplied mathematical notation: do NOT guess or change equalities into inequalities.
If a change is necessary, return the SMALLEST exact replacement(s) in NEUTRAL that fix MEANING.
Each patch: before (an exact unique substring of NEUTRAL), after (replacement), reason (specific meaning correction).
Never replace an entire paragraph. Do not restore technical vocabulary if the neutral phrase already means the same thing.
For an omitted proposition, replace a short unique anchor with that anchor plus a short neutral sentence containing the missing claim.
Return ONLY a JSON array, one per id: {id, verdict: aligned|repair|source_uncertain, patches: [], notes: []}.
Aligned has no patches. Source_uncertain is only for an actually unreadable or contradictory HUMAN source, not references to other sections/tables.
Preserve all existing neutral wording outside the supplied small patches. Keep the human text separate; do not return it as the rewrite.
'''


def prepare():
    m.check()
    if RUN.exists():raise ValueError('refuse_overwrite_minimal_review')
    targets=m.effective_targets();drafts={r['id']:r for r in m.lines(m.RUN/'generated_drafts.private.jsonl')}
    jobs=[]
    for start in range(0,len(targets),54):
        chunk=targets[start:start+54];ids={f'M{start+i+1:03d}':r['id'] for i,r in enumerate(chunk)}
        rows=[dict(id=sid,HUMAN=next(r['target'] for r in chunk if r['id']==rid),NEUTRAL=drafts[rid]['plain_english']) for sid,rid in ids.items()]
        prompt=PROMPT+'\nPAIRS:\n'+json.dumps(rows,ensure_ascii=False)
        jobs.append(dict(index=len(jobs)+1,ids=list(ids),id_map=ids,prompt=prompt,prompt_sha256=m.sha(prompt)))
    RUN.mkdir();(RUN/'verification').mkdir()
    config=m.read(m.RUN/'config.json');config['temperature']=.2;config['max_output_tokens']=16384
    m.write(RUN/'config.json',config);m.write_lines(RUN/'verification/jobs.private.jsonl',jobs)
    m.write(RUN/'manifest.json',dict(builder_sha256=m.digest(Path(__file__)),target_repairs_sha256=m.digest(m.RUN/'target_repairs.private.jsonl'),
        drafts_sha256=m.digest(m.RUN/'generated_drafts.private.jsonl'),files={str(p.relative_to(RUN)):m.digest(p) for p in RUN.rglob('*') if p.is_file()}))
    print('Prepared four minimal meaning-patch reviews; no API calls.')


def parse(text,ids):
    rows=json.loads(text)
    if not isinstance(rows,list) or len(rows)!=len(ids) or {r['id'] for r in rows}!=set(ids):raise ValueError('review_coverage')
    for r in rows:
        if r['verdict'] not in {'aligned','repair','source_uncertain'} or not isinstance(r['patches'],list):raise ValueError('review_schema')
        if r['verdict']=='aligned' and r['patches']:raise ValueError('aligned_patch')
    return rows


def run():
    manifest=m.read(RUN/'manifest.json')
    for name,h in manifest['files'].items():
        if m.digest(RUN/name)!=h:raise ValueError('review_inputs_changed')
    for job in m.lines(RUN/'verification/jobs.private.jsonl'):
        path=RUN/'verification'/f"response_{job['index']:02d}.private.json"
        if path.exists():
            r=m.read(path)
            if r['status']=='generated':parse(r['text'],job['ids']);continue
            print('Prior failed request retained; no fallback.');return
        print(f"Minimal semantic review {job['index']}/4",flush=True)
        result=subprocess.run(['Rscript',str(Path(__file__).with_name('expand_stage_500.R')),str(RUN),'verification',str(job['index']),str(path)],capture_output=True,timeout=620)
        if result.returncode or not path.exists() or m.read(path)['status']!='generated':
            print('Paused on provider/quota failure; all evidence retained; no paid fallback.',flush=True);return
        parse(m.read(path)['text'],job['ids']);print('Saved review.',flush=True)
        if job['index']<4:time.sleep(21)


def apply_patches(source,patches):
    result=source
    for p in patches:
        before=p['before'];after=p['after']
        if not isinstance(before,str) or not before or result.count(before)!=1:raise ValueError('patch_anchor_not_unique')
        if len(before.split())>max(25,len(source.split())*.6):raise ValueError('patch_reverts_too_much_wording')
        result=result.replace(before,after,1)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    {'prepare':prepare,'run':run}[a.action]()
