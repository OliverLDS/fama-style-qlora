"""Prepare ordinary rewrite prompts by default, with explicit optional edit plans.

CPU-only prompt preparation. No model execution, automatic claim extraction or
learned structural mapping. Historical experiment/training prompts stay frozen.
"""
import argparse
import json
from pathlib import Path
import re
from fama_style.rhetoric import readable_packing as readable
from fama_style.rhetoric import paper_plan_comparison as comparison
from fama_style.rhetoric.experiment import fresh, write_json
from fama_style.research.claims import sha

PROTOCOL = 'economics_rewrite_prototype_v1'
MODES = ('ordinary', 'planned')
EPISTEMIC = {'finding', 'hypothesis', 'definition', 'recommendation', 'assertion', 'expectation', 'unresolved'}
PROTECTED = {'actor', 'quantity', 'condition', 'uncertainty', 'negation', 'scope', 'comparison', 'attribution', 'causal_direction'}
RELATIONS = {'inference', 'conditional', 'causal', 'contrast', 'alternative', 'presentational'}


def fields(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys): raise ValueError('Unexpected or missing fields')


def refs(value, known):
    if not isinstance(value, list) or not value or any(not isinstance(c, str) for c in value) or len(value) != len(set(value)) or not set(value) <= known:
        raise ValueError('Unknown, empty or duplicate identifiers')


def spans(source, value):
    if not isinstance(value, list) or not value: raise ValueError('Empty source evidence')
    for s in value: fields(s, ('start', 'end', 'quote'))
    readable.quoted(source, value)


def validate_evidence(source, evidence):
    fields(evidence, ('source_units', 'claim_index', 'source_relations'))
    units = evidence['source_units']; claims = evidence['claim_index']; relations = evidence['source_relations']
    if not isinstance(units, list) or not units or not isinstance(claims, list) or not claims or not isinstance(relations, list):
        raise ValueError('Evidence needs source units, claims and a relation list')
    cursor = 0; bounds = []
    for i, u in enumerate(units, 1):
        fields(u, ('unit_id', 'text'))
        if u['unit_id'] != 'S' + str(i) or not isinstance(u['text'], str) or not u['text'].strip(): raise ValueError('Invalid ordered source unit')
        start = source.find(u['text'], cursor)
        if start < 0 or source[cursor:start].strip(): raise ValueError('Source units omit or change source text')
        cursor = start + len(u['text']); bounds.append((start, cursor))
    if source[cursor:].strip(): raise ValueError('Source units omit trailing text')
    known = set(); groups = {u['unit_id']: [] for u in units}
    for c in claims:
        fields(c, ('claim_id', 'composite', 'epistemic_status', 'source_spans', 'protected'))
        cid = c['claim_id']
        if not isinstance(cid, str) or not re.fullmatch(r'c[1-9][0-9]*', cid) or cid in known: raise ValueError('Invalid claim identity')
        if type(c['composite']) is not bool or c['epistemic_status'] not in EPISTEMIC or not isinstance(c['protected'], list): raise ValueError('Invalid claim annotation')
        known.add(cid); spans(source, c['source_spans'])
        containing = [i for i, (a, b) in enumerate(bounds) if all(a <= s['start'] < s['end'] <= b for s in c['source_spans'])]
        if len(containing) != 1: raise ValueError('Whole claim must be bound within one source unit')
        groups[units[containing[0]]['unit_id']].append(cid)
        for p in c['protected']:
            fields(p, ('kind', 'source_spans'))
            if p['kind'] not in PROTECTED: raise ValueError('Unknown semantic protection')
            spans(source, p['source_spans'])
    if any(not g for g in groups.values()): raise ValueError('Unbound source unit')
    relation_ids = set()
    for r in relations:
        fields(r, ('relation_id', 'kind', 'from_claim_refs', 'to_claim_refs', 'source_evidence', 'certainty'))
        if not isinstance(r['relation_id'], str) or not r['relation_id'] or r['relation_id'] in relation_ids: raise ValueError('Invalid relation identity')
        relation_ids.add(r['relation_id'])
        if r['kind'] not in RELATIONS or r['certainty'] not in {'explicit', 'interpretive'}: raise ValueError('Unknown relation annotation')
        refs(r['from_claim_refs'], known); refs(r['to_claim_refs'], known)
        if not isinstance(r['source_evidence'], list) or not r['source_evidence']: raise ValueError('Missing relation evidence')
        for s in r['source_evidence']:
            fields(s, ('quote', 'occurrence'))
            if type(s['occurrence']) is not int or s['occurrence'] < 1: raise ValueError('Invalid evidence occurrence')
        readable.parent.edits.base.resolve(source, r['source_evidence'])
    return groups


def validate_plan(source, evidence, plan):
    groups = validate_evidence(source, evidence)
    if not isinstance(plan, list) or not plan: raise ValueError('Provide an explicit edit program')
    ordered_units = list(groups); consumed = []; all_claims = []; claims = {c['claim_id']: c for c in evidence['claim_index']}
    for op in plan:
        fields(op, ('op', 'source_unit_ids', 'output_slots'))
        code = op['op']; ids = op['source_unit_ids']; slots = op['output_slots']
        refs(ids, set(groups))
        if ids != ordered_units[len(consumed):len(consumed) + len(ids)]: raise ValueError('Source units must be consumed once in order')
        consumed.extend(ids)
        if not isinstance(slots, list) or not slots: raise ValueError('Empty output slots')
        flat = []
        for s in slots:
            fields(s, ('claim_refs',)); refs(s['claim_refs'], set(claims)); flat.extend(s['claim_refs'])
        if sorted(flat) != sorted(c for uid in ids for c in groups[uid]): raise ValueError('Operation changes whole-claim coverage')
        if code == 'KEEP':
            if len(ids) != 1 or len(slots) != 1: raise ValueError('KEEP requires one source and one output unit')
        elif code == 'MERGE':
            if len(ids) < 2 or len(slots) != 1: raise ValueError('MERGE requires multiple sources and one output unit')
        elif code == 'SPLIT':
            if len(ids) != 1 or len(slots) < 2: raise ValueError('SPLIT requires one source and multiple output units')
            for i, cid in enumerate(flat):
                for other in flat[i + 1:]:
                    if any(a['start'] < b['end'] and b['start'] < a['end'] for a in claims[cid]['source_spans'] for b in claims[other]['source_spans']):
                        raise ValueError('Split needs distinct grounded claims; composite subclaims cannot be guessed')
            starts = [min(s['start'] for s in claims[c]['source_spans']) for c in flat]
            if starts != sorted(starts): raise ValueError('Split must preserve source claim order')
        else: raise ValueError('Unknown edit operation')
        all_claims.extend(flat)
    if consumed != ordered_units or sorted(all_claims) != sorted(claims): raise ValueError('Incomplete source/claim coverage')


def make_prompt(source, mode='ordinary', evidence=None, plan=None):
    if not isinstance(source, str) or not source.strip(): raise ValueError('Provide nonempty source text')
    if mode not in MODES: raise ValueError('Unknown rewrite mode')
    if mode == 'ordinary' and plan is not None: raise ValueError('Use planned mode explicitly when supplying a plan')
    if mode == 'planned' and (evidence is None or plan is None): raise ValueError('Planned mode requires source evidence and an explicit plan')
    if evidence is not None:
        validate_evidence(source, evidence)
        payload = {'source': source, **evidence}
        prompt = readable.readable_prefix(payload) + comparison.COMMON_TAIL
    else:
        prompt = readable.parent.COMMON + '\nFull source:\n' + source
    if mode == 'planned':
        validate_plan(source, evidence, plan)
        prompt += '\nSentence-packing edit plan:\n' + readable.readable_plan({**payload, 'edits': plan})
    return prompt + '\nReturn only the rewritten paragraph.'


def prepare(source_path, output, mode='ordinary', evidence_path=None, plan_path=None):
    source = source_path.read_text(encoding='utf-8')
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path else None
    plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path else None
    prompt = make_prompt(source, mode, evidence, plan)
    fresh(output); (output / 'prompt.txt').write_text(prompt, encoding='utf-8')
    write_json(output / 'request.json', {'prompt': prompt, 'prompt_sha256': sha(prompt)})
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'mode': mode, 'source_sha256': sha(source), 'prompt_sha256': sha(prompt),
               'evidence_supplied': evidence is not None, 'explicit_plan_supplied': plan is not None,
               'input_hashes': {k: sha(p.read_text(encoding='utf-8')) for k, p in [('source', source_path), ('evidence', evidence_path), ('plan', plan_path)] if p},
               'model_calls': 0, 'training_called': False, 'semantic_preservation_verified': False, 'author_style_accuracy': None,
               'scope': 'Prompt preparation only. Plain-source mode is not the evaluated shared-evidence baseline. Evidence annotations are fallible; no learned mapper or automatic extraction.'})
    print('Prepared ' + mode + ' rewrite prompt. No model call, credential access or training.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-file', type=Path, required=True); p.add_argument('--mode', choices=MODES, default='ordinary')
    p.add_argument('--evidence-file', type=Path); p.add_argument('--plan-file', type=Path); p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv); prepare(a.source_file, a.output_dir, a.mode, a.evidence_file, a.plan_file)


if __name__ == '__main__': main()
