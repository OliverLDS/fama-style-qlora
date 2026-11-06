"""Recover existing pairs, generate missing inputs on free Gemini, freeze 500; no training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_runs/data_scaling_v1/stage_0500_expansion_v1'
OUT = ROOT / 'data/data_scaling_v1/stage_0500_expansion_v1'
GROUPS = ('Fama_coauthors', 'Shiller', 'Holmstrom', 'Baker_Wurgler')
MODEL = 'gemini-3-flash-preview'
GEN = '''Create meaning-preserving neutral-English inputs for a supervised rewriting dataset.
For EACH human paragraph, return its id and plain_english. Use ordinary, clear English for an economically literate reader.
This is a FULL PARAGRAPH REWRITE, NOT A SUMMARY. Preserve every supporting claim, definition, example, comparison,
actor, condition, exception, qualification, attribution, uncertainty, negation and causal direction.
Keep ALL quantities and citations, including page/chapter details, exactly as written. An order of magnitude is not merely 'larger'.
Do not omit mathematical/statistical definitions. 'Be wary' is not 'avoid'; 'may' is not 'does'.
Do not add explanations or economic mechanisms. Simplify sentence structure and vocabulary without losing information.
Do not merely replace connectives with synonyms. Where safe simplification is impossible, retain the wording.
Before returning, compare every sentence's claims against the input and restore any omitted details.
Return ONLY a JSON array of objects with exactly id and plain_english, one per supplied paragraph, with no Markdown.
'''
VERIFY = '''Compare each neutral rewrite with its original human paragraph in BOTH directions.
Check all claims, actors, definitions, examples, citations (including page details), quantities, comparisons,
conditions, exceptions, modality, negation, attribution and causal direction. This is a rewrite, NOT a summary.
If the rewrite omits or changes anything, REPAIR THE REWRITE to preserve the original paragraph's explicit meaning.
Do not alter the original, invent facts, or guess damaged source notation. Prefer a complete faithful repair to rejecting an example.
External table/section references do not require outside context if the explicit claims can simply be retained.
Return ONLY a JSON array, one object per supplied id, with exactly:
id, verdict (aligned, repaired, or source_uncertain), plain_english (the complete final rewrite), notes (array of specific changes or uncertainty).
For aligned, copy the supplied rewrite exactly. For repaired, supply the corrected complete neutral rewrite.
Use source_uncertain ONLY for actual damaged/ambiguous original wording preventing faithful rewriting.
Keep every numeral/citation as written in the human original. Restore all definitions and qualifications.
'''


def sha(text): return hashlib.sha256(text.encode()).hexdigest()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def write_lines(path, rows): path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
def canonical(text): return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
def numbers(text): return Counter(re.findall(r'\d+(?:\.\d+)?', text))


def interleave(rows):
    papers = defaultdict(list)
    for row in rows: papers[row['paper_id']].append(row)
    for paper in papers: papers[paper].sort(key=lambda r: sha('stage500|'+r['id']))
    order = sorted(papers, key=lambda p: sha('stage500-paper|'+p)); result = []; depth = 0
    while any(depth < len(papers[p]) for p in order):
        result.extend(papers[p][depth] for p in order if depth < len(papers[p])); depth += 1
    return result


def normalize_pair(row, source, target, origin, approval):
    return dict(row, plain_english=source, original_paragraph=target, source=source, target=target,
        paragraph_id=row['id'], parent_target_id=row['parent_id'], source_sha256=sha(source), target_sha256=sha(target),
        split='training', training_ready=True,
        quality=dict(text_pair_status='automated_pilot_approved', semantic_review=approval,
            independent_human_verification=False, full_visual_transcription_verified=False,
            structural_annotations_status='not_qualified_for_structural_training'),
        provenance=origin)


def prepare():
    if RUN.exists(): raise ValueError('refuse_overwrite_expansion')
    base = ROOT/'data/data_scaling_v1/stage_0200_salvage_v1'
    for name, h in read(base/'manifest.json')['outputs'].items():
        if digest(base/name) != h: raise ValueError('base_changed')
    prefix = lines(base/'training.private.jsonl'); spare = lines(base/'remaining_approved.private.jsonl')
    if len(prefix) != 200 or len(spare) != 10: raise ValueError('base_counts')
    seen = {canonical(r['target']) for r in prefix+spare}; audit = []; recovered = []
    oldfile = ROOT/'data/fama_corpus_v1/train.jsonl'; oldreport = read(oldfile.parent/'build_report.json')
    if digest(oldfile) != oldreport['files']['train.jsonl']: raise ValueError('legacy_changed')
    legacy = []
    for row in lines(oldfile):
        if canonical(row['original_paragraph']) in seen: continue
        a,b = row['plain_english'],row['original_paragraph']; q = row['quality']
        reasons = []
        if row['split'] != 'train' or q.get('semantic_reviewer') != 'assistant' or q.get('source_review_flag'): reasons.append('prior_source_or_semantic_uncertainty')
        if numbers(a) != numbers(b): reasons.append('numeric_recheck')
        if any(unicodedata.category(c)=='Co' for c in a+b): reasons.append('corrupt_glyph')
        if not 40 <= len(b.split()) <= 250: reasons.append('length')
        audit.append(dict(parent_id=row['id'],category='legacy_screen',reasons=reasons))
        if not reasons: legacy.append(row)
    for row in interleave(legacy)[:75]:
        result = normalize_pair(dict(id=row['id']+'::stage500_requalified_v1', parent_id=row['id'],
            paper_id=row['paper_id'], style_group='Fama_coauthors', authors=row['authors']),
            row['plain_english'], row['original_paragraph'],
            dict(parent_file=str(oldfile.relative_to(ROOT)), parent_file_sha256=digest(oldfile), parent_id=row['id'],
                 legacy_quality=row['quality'], legacy_provenance=row['provenance'],
                 original_generation_model='historical annotation model not re-established; not relabeled GPT-6'),
            'Prior explicit assistant semantic acceptance reused; new hash, numeric, split and source-flag checks; not a fresh independent semantic review')
        recovered.append(result); seen.add(canonical(result['target']))
        audit.append(dict(parent_id=row['id'],category='legacy_requalified',source_sha256=result['source_sha256'],target_sha256=result['target_sha256'],
                          target_changed=False,source_changed=False,evidence='fama_corpus_v1 frozen training export and build_report'))
    if len(recovered) != 75: raise ValueError('insufficient_legacy_Fama')
    roles = defaultdict(set); annotated = set(); source_files = []
    for name in ['existing_collection_package_v2','new_collection_package_v1']:
        p = ROOT/'research_runs/author_pair_expansion_v1'/name/'selection.private.json';source_files.append(p)
        for r in read(p): roles[r['paper_id']].add(r['role']);annotated.add(r['paragraph_id'])
    prior_file=ROOT/'research_runs/multi_author_milestone_v1/package_v1/datasets/train.provisional.private.jsonl'
    source_files.append(prior_file)
    prior_inputs={canonical(r['target']):r for r in lines(prior_file) if r['split']=='training'}
    candidates = []; candidate_seen = set(); registries={}
    for name in ['author_style_corpus_v3_existing','author_style_corpus_v3']:
        p = ROOT/'data'/name/'paragraphs.private.jsonl'; source_files.append(p)
        registry=p.parent/'source_registry.private.json';source_files.append(registry)
        registries.update({r['paper_id']:r for r in read(registry)['papers']})
        for r in lines(p):
            target = r['original_paragraph']; c = canonical(target)
            prior=prior_inputs.get(c)
            earlier_training=prior is not None and prior['paper_id']==r['paper_id']
            if (roles.get(r['paper_id']) != {'training'} and not earlier_training) or r['paragraph_id'] in annotated or r['style_group']=='Fama_coauthors':continue
            if 'development' in roles.get(r['paper_id'],set()):continue
            if c in seen or c in candidate_seen:continue
            if not all('native' in m for m in r['extraction_methods']) or r['unresolved_hyphens'] or not 40 <= len(target.split()) <= 250:continue
            if sha(target) != r['original_sha256']:raise ValueError('candidate_text_changed')
            candidate_seen.add(c)
            candidates.append(dict(id=r['paragraph_id'], parent_id=r['paragraph_id'], paper_id=r['paper_id'],
                authors=r['authors'],style_group=r['style_group'],target=target,target_sha256=sha(target),
                pdf_sha256=r['paper_pdf_sha256'],line_refs=r['line_refs'],page_location=r['page_location'],
                quality_flags=r['quality_flags'],source_file=str(p.relative_to(ROOT)),source_file_sha256=digest(p),
                previous_source=prior['source'] if earlier_training else None,
                previous_source_file=str(prior_file.relative_to(ROOT)) if earlier_training else None))
    reused = prefix+spare+recovered; needs = {g:125-sum(r['style_group']==g for r in reused) for g in GROUPS}
    selected=[]
    for group in GROUPS:
        pool=interleave([r for r in candidates if r['style_group']==group])
        pool.sort(key=lambda r:r['previous_source'] is None)  # Recover collected annotations first.
        if len(pool)<needs[group]:raise ValueError('candidate_shortage_'+group)
        selected.extend(pool[:needs[group]])
    if len(selected)!=215:raise ValueError('expected_215_missing_inputs')
    # Mix groups within requests so one provider failure cannot silently remove a group.
    selected.sort(key=lambda r:sha('stage500-batch|'+r['id']))
    pdf_checks={}
    for row in selected:
        paper=row['paper_id']
        if paper not in pdf_checks:
            pdf_checks[paper]=digest(ROOT/registries[paper]['pdf'])
        if pdf_checks[paper]!=row['pdf_sha256']:raise ValueError('source_PDF_changed')
    RUN.mkdir(parents=True);(RUN/'generation').mkdir();(RUN/'verification').mkdir()
    write_lines(RUN/'reused.private.jsonl',reused);write_lines(RUN/'missing_inputs.private.jsonl',selected)
    write_lines(RUN/'requalification_audit.private.jsonl',audit)
    missing=[r for r in selected if r['previous_source'] is None]
    seeds=[dict(id=r['id'],plain_english=r['previous_source'],generation_response_sha256=digest(prior_file),
                generator='GPT-6 historical paired annotation; session provenance retained in original release',
                prior_file=r['previous_source_file']) for r in selected if r['previous_source'] is not None]
    write_lines(RUN/'recovered_drafts.private.jsonl',seeds)
    chunks=[missing[:4]]+[missing[i:i+35] for i in range(4,len(missing),35)]
    jobs=[]
    for i,chunk in enumerate(chunks):
        prompt=GEN+'\nINPUTS:\n'+json.dumps([dict(id=r['id'],original_paragraph=r['target']) for r in chunk],ensure_ascii=False)
        jobs.append(dict(index=i+1,ids=[r['id'] for r in chunk],prompt=prompt,prompt_sha256=sha(prompt)))
    write_lines(RUN/'generation/jobs.private.jsonl',jobs)
    write(RUN/'config.json',dict(protocol='stage_0500_expansion_v1',model=MODEL,free_tier_only=True,
        free_project_confirmed_by_user=True,paid_fallback=False,training_authorized=False,
        max_output_tokens=32768,temperature=.5,top_p=1,top_k=40,
        prefix_sha256=digest(base/'training.private.jsonl'),legacy_training_sha256=digest(oldfile),
        generator_model_change_explicit=True,existing_200_prefix_unchanged=True,
        reused=285,drafts_to_review=215,missing_inputs=len(missing),prior_draft_inputs_recovered=len(seeds),
        source_pdf_hashes_verified=pdf_checks,source_files={str(p.relative_to(ROOT)):digest(p) for p in source_files},
        runner_sha256=digest(Path(__file__)),adapter_sha256=digest(Path(__file__).with_suffix('.R'))))
    write(RUN/'preparation_manifest.json',{'files':{str(p.relative_to(RUN)):digest(p) for p in RUN.rglob('*') if p.is_file()}})
    print(f'Prepared: 285 reused/requalified pairs; {len(seeds)} recovered draft inputs; {len(missing)} missing inputs; {len(jobs)} generation jobs. No API calls.')


def check():
    config=read(RUN/'config.json')
    for rel,h in read(RUN/'preparation_manifest.json')['files'].items():
        if digest(RUN/rel)!=h:raise ValueError('prepared_input_changed')
    if (not config['free_tier_only'] or config['paid_fallback'] or not config['free_project_confirmed_by_user']
            or config['model']!=MODEL or config['runner_sha256']!=digest(Path(__file__))
            or config['adapter_sha256']!=digest(Path(__file__).with_suffix('.R'))):raise ValueError('free_only_contract')
    return config


def parse_array(text, expected):
    stripped=text.strip()
    if stripped.startswith('```'):
        stripped=re.sub(r'^```(?:json)?\s*','',stripped);stripped=re.sub(r'\s*```$','',stripped)
    result=json.loads(stripped)
    if not isinstance(result,list) or len(result)!=len(expected):raise ValueError('response_coverage')
    ids=[r['id'] for r in result]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('response_ids')
    for r in result:
        if not isinstance(r.get('plain_english'),str) or not r['plain_english'].strip():raise ValueError('empty_rewrite')
    return result


def effective_targets():
    """Apply only separately recorded PDF-checked child repairs; keep frozen extraction intact."""
    rows=lines(RUN/'missing_inputs.private.jsonl');by={r['id']:r for r in rows}
    path=RUN/'target_repairs.private.jsonl'
    if not path.exists():return rows
    for repair in lines(path):
        row=by[repair['id']]
        if sha(row['target'])!=repair['before_sha256'] or sha(repair['after'])!=repair['after_sha256']:
            raise ValueError('target_repair_binding')
        for rel,h in repair['evidence'].items():
            if digest(RUN/rel)!=h:raise ValueError('source_evidence_changed')
        row['extraction_target_sha256']=row['target_sha256']
        row['target']=repair['after'];row['target_sha256']=repair['after_sha256']
        row['target_repair']=dict(audit_sha256=digest(path),reason=repair['reason'],evidence=repair['evidence'])
    return rows


def response_text(path):
    text=read(path)['text'];audit=RUN/'response_id_repairs.private.jsonl'
    if audit.exists():
        for repair in lines(audit):
            if str(path.relative_to(RUN))!=repair['response_relative_path']:continue
            if digest(path)!=repair['response_sha256']:raise ValueError('response_ID_repair_binding')
            if 'object_index' in repair:
                parsed=json.loads(text);item=parsed[repair['object_index']]
                if item['id']!=repair['from_id'] or sha(item['plain_english'])!=repair['text_sha256']:
                    raise ValueError('indexed_ID_repair_binding')
                item['id']=repair['to_id'];text=json.dumps(parsed,ensure_ascii=False);continue
            old=json.dumps(repair['from_id']);new=json.dumps(repair['to_id'])
            if text.count(old)!=1 or new in text:raise ValueError('ambiguous_ID_repair')
            text=text.replace(old,new)
    return text


def call_jobs(phase):
    check();folder=RUN/phase;jobs=lines(folder/'jobs.private.jsonl')
    for job in jobs:
        path=folder/f"response_{job['index']:02d}.private.json"
        if path.exists():
            r=read(path)
            if r['prompt_sha256']!=job['prompt_sha256']:raise ValueError('resume_prompt_changed')
            if r['status']=='generated':parse_array(response_text(path),job['ids']);continue
            print('Existing failed response retained; no automatic retry or fallback.');return
        print(f"{phase}: request {job['index']}/{len(jobs)} ({len(job['ids'])} paragraphs)",flush=True)
        try:
            process=subprocess.run(['Rscript',str(Path(__file__).with_suffix('.R')),str(RUN),phase,str(job['index']),str(path)],capture_output=True,timeout=620)
        except subprocess.TimeoutExpired:
            print('Paused: request deadline; inspect retained evidence before retry.',flush=True);return
        if process.returncode or not path.exists() or read(path)['status']!='generated':
            print('Paused: provider/completion failure. No paid fallback or automatic retry.',flush=True);return
        r=read(path)
        try:parse_array(response_text(path),job['ids'])
        except (ValueError,KeyError,TypeError):
            print('Paused: batch schema/coverage issue; entire raw response retained.',flush=True);return
        print(f"{phase}: saved {len(job['ids'])} paragraphs",flush=True)
        if job['index']<len(jobs):time.sleep(21)
    print(phase+' complete.',flush=True)


def prepare_verification():
    check();jobs=lines(RUN/'generation/jobs.private.jsonl');by={r['id']:r for r in lines(RUN/'recovered_drafts.private.jsonl')}
    for job in jobs:
        path=RUN/'generation'/f"response_{job['index']:02d}.private.json";r=read(path)
        if r['status']!='generated':raise ValueError('generation_incomplete')
        for item in parse_array(response_text(path),job['ids']):by[item['id']]=dict(item,generation_response_sha256=digest(path),generator=MODEL,
                                                                     thinking_level=r.get('thinking_level','provider_default'))
    selected=effective_targets()
    id_map={f'V{i+1:03d}':r['id'] for i,r in enumerate(selected)}
    rows=[dict(id=f'V{i+1:03d}',original_paragraph=r['target'],plain_english=by[r['id']]['plain_english']) for i,r in enumerate(selected)]
    result=[]
    for i in range(0,len(rows),54):
        chunk=rows[i:i+54];prompt=VERIFY+'\nPAIRS:\n'+json.dumps(chunk,ensure_ascii=False)
        result.append(dict(index=len(result)+1,ids=[r['id'] for r in chunk],id_map={r['id']:id_map[r['id']] for r in chunk},prompt=prompt,prompt_sha256=sha(prompt)))
    path=RUN/'verification/jobs.private.jsonl'
    if path.exists():raise ValueError('verification_already_frozen')
    write_lines(path,result);write_lines(RUN/'generated_drafts.private.jsonl',[dict(by[r['id']],target_sha256=r['target_sha256']) for r in selected])
    write(RUN/'verification/preparation_manifest.json',{'jobs_sha256':digest(path),'drafts_sha256':digest(RUN/'generated_drafts.private.jsonl')})
    print('Prepared four separate semantic verification/repair jobs for all 215 drafts.')


def assemble():
    # Full-paragraph verifier proposals are evidence only: they often restore
    # the human target verbatim and would erase the rewriting training signal.
    subprocess.run(['python3', str(Path(__file__).with_name('finalize_stage_500.py'))], check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','generate','prepare-verification','verify','assemble']);a=p.parse_args()
    {'prepare':prepare,'check':check,'generate':lambda:call_jobs('generation'),
     'prepare-verification':prepare_verification,'verify':lambda:call_jobs('verification'),'assemble':assemble}[a.action]()
