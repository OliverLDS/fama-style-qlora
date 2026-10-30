"""Prepare a private, development-only human-prose oracle realization check."""
import argparse
import hashlib
import json
from pathlib import Path

from .author_control_replication import verify_files
from .human_prose_review import LEVELS
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from .author_comparison_corpus import file_sha

PROTOCOL = 'human_prose_realization_v1'
PROMPT = '''Rewrite each supplied source as one clear, compact paragraph of academic economics prose. Preserve its substantive claims, actors, quantities, causal direction, negation, uncertainty and conditions. Improve organization and expression where useful; do not add facts, explanations or economic relationships. Resolve nothing using outside context. Return only the rewritten paragraph in the requested JSONL field.

The claim evidence quotes are excerpts from the source, not replacement prose. If a presentation_plan is supplied, use it as optional organizational guidance: its claim groups suggest sentence organization and its controls describe presentation features. Ignore a suggestion if it requires changing meaning or makes the text awkward. No particular author is requested. Do not optimize for a detector or force a wording-distance target.

Return one JSON object per input, with exactly job_id and output fields. Do not include commentary or Markdown fences. Do not ask for or retrieve the reference paragraph.''' 


def payload(row, catalog, planned):
    """Project onto source text/evidence and finite, checked reference structure.

    Never export propositions, target text/evidence, explanations or author IDs.
    The plan remains reference-informed even though reference wording is removed.
    """
    a = row['annotation']
    claims = {c['claim_id']: c for c in a['semantic_ledger']['claims']}
    result = {'source': a['plain_english'], 'claim_evidence': [
        {'claim_id': cid, 'source_quotes': [e['quote'] for e in c['rewrite_evidence']]}
        for cid, c in claims.items()]}
    if planned:
        sentences = {s['sentence_id']: set(s['claim_ids']) for s in a['target_schema']['sentences']}
        controls = []
        for field in LEVELS:
            for obs in a['target_schema'][field]:
                category = catalog.get(obs['canonical_id'])
                bound = set().union(*(sentences.get(s, set()) for s in obs['sentence_ids'])) if obs['sentence_ids'] else set()
                if not (category and category['control_status'] == 'candidate_presentation_control'
                        and obs['assignment_status'] == 'assigned' and obs['value'] == 'present'
                        and obs['claim_ids'] and obs['sentence_ids']
                        and set(obs['claim_ids']) <= bound <= set(claims)):
                    continue
                controls.append({'canonical_id': obs['canonical_id'], 'feature': category['name'],
                    'definition': category['definition'], 'claim_ids': obs['claim_ids'],
                    'sentence_ids': obs['sentence_ids']})
        result['presentation_plan'] = {
            'sentence_claim_groups': [s['claim_ids'] for s in a['target_schema']['sentences']],
            'controls': controls}
    return result


def run(audit, package, output):
    verify_files(audit)
    verify_files(package)
    records = [json.loads(line) for line in (audit/'audits.private.jsonl').read_text().splitlines()]
    catalog = {c['canonical_id']: c for c in json.loads((package/'common/taxonomy_catalog.compact.json').read_text())['canonical_items']}
    eligible = [r for r in records if r['split']=='development' and r['schema_valid']
        and not r['audit']['errors'] and not r['audit']['program_errors']
        and not r['audit']['metrics']['normalized_token_copy']
        and all(c['alignment_status']=='preserved' for c in r['annotation']['semantic_ledger']['claims'])]
    papers = sorted(set(r['paper_id'] for r in eligible))
    selected=[]
    for paper in papers:
        candidates = [r for r in eligible if r['paper_id']==paper]
        # Prefer fewer context flags, then deterministic ID order; no output-based selection.
        candidates.sort(key=lambda r: (r['annotation']['source_quality']['needs_external_context'],
            r['annotation']['needs_review'], r['paragraph_id']))
        if len(candidates)<2: raise ValueError('insufficient_development_candidates')
        selected.extend(candidates[:2])
    if len(papers)!=4 or len(selected)!=8: raise ValueError('expected_four_development_papers')
    jobs=[[],[]];key=[]
    for i,row in enumerate(selected):
        for planned in (False,True):
            job_id=hashlib.sha256((PROTOCOL+row['paragraph_id']+str(planned)).encode()).hexdigest()[:16]
            batch=(i+int(planned))%2
            jobs[batch].append({'job_id':job_id,**payload(row,catalog,planned)})
            key.append({'job_id':job_id,'paragraph_id':row['paragraph_id'],'paper_id':row['paper_id'],
                'condition':'oracle_presentation' if planned else 'ordinary','batch':batch+1,
                'source_needs_external_context':row['annotation']['source_quality']['needs_external_context']})
    fresh(output)
    for i,batch in enumerate(jobs,1):
        directory=output/('batch_%02d'%i);directory.mkdir()
        write_lines(directory/'inputs.jsonl',batch)
        (directory/'TASK.md').write_text(PROMPT+'\n\nUse the attached inputs.jsonl. Save responses_%02d.jsonl.\n'%i)
    write_json(output/'private_key.json',key)
    write_json(output/'summary.json',{'protocol':PROTOCOL,'examples':8,'papers':4,'generations_requested':16,
        'status':'prepared_not_executed','reserved_rows_used':0,'training_started':False,
        'conditions':['ordinary','reference_informed_oracle_presentation'],
        'limitations':['Reference-informed plans are not deployable source-only mappings.',
            'Same-model annotation and realization are not independent semantic verification.',
            'Cross-chat retrieval and hidden decoding settings are uncontrolled.',
            'The sample and taxonomy do not establish author-distinctive style.']})
    (output/'responses').mkdir()
    (output/'START_HERE.md').write_text('''# Small human-prose realization diagnostic

Two batches, eight jobs each: ordinary rewriting versus optional oracle presentation guidance on the same eight development sources. Use GPT-6, High thinking effort for both. Keep the first responses unchanged. Separate chats are preferable, but they do not establish independence from prior context or retrieval. Record any changed model/settings and shared context. Do not attach annotations, originals, private_key.json or previous answers.

For each batch, paste TASK.md and attach only its inputs.jsonl. Save responses_01.jsonl and responses_02.jsonl in this package's responses/ folder. Do not edit outputs to improve them. No training or Kaggle execution is required.

Plans use reference sentence groups and finite presentation categories without reference prose. This is an oracle feasibility diagnostic, not an author-style or source-only mapping result. The two conditions share source and source-quoted claim evidence; only the plan differs. Some paragraphs retain external-context flags. Future review must separate observed claim drift, copying, readability and plan adherence; none of these alone proves distinctive style or human authorship.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'audit_manifest_sha256':file_sha(audit/'manifest.json'),
        'code_sha256':file_sha(Path(__file__)),'taxonomy_sha256':file_sha(package/'common/taxonomy_catalog.compact.json'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','package','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(run(a.audit,a.package,a.output))


if __name__=='__main__': main()
