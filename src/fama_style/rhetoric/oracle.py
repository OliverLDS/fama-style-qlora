"""Content-free oracle templates, balanced prompts and automatic diagnostic scoring."""
import copy
import difflib
import json
import statistics

from fama_style.prompting.corpus import make_prompt
from fama_style.research.claims import sha, verify, risk_flags, tokens
from fama_style.research.signal import overlap, distance
from .extract import extract, flatten
from .schema import validate, validate_links

CONDITIONS = ('A_plain', 'B_source_structure', 'C_oracle_target_structure')
PROTOCOL = 'oracle_rhetoric_v1'
PRESENTATION_KEYS = {'organization', 'sentence_count', 'syntax', 'condition_position', 'sentence_length',
                     'author_voice', 'semicolon', 'parenthesis', 'nominalization'}


def structural_template(structure):
    """Copy only validated finite labels; omit hashes, claims and reference spans."""
    validate(structure)
    return {'paragraph': {'organization': structure['paragraph']['organization'],
                          'sentence_count': structure['paragraph']['sentence_count'],
                          'move_roles': [item['role'] for item in structure['paragraph']['moves']]},
            'sentences': [{key: item[key] for key in ('syntax', 'condition_position', 'voice', 'length')}
                          for item in structure['sentences']],
            'language': {key: value for key,value in structure['language'].items() if key not in ('hedging','citation')}}


def adherence(actual, desired):
    """Score finite feature buckets and the requested paragraph/sentence sequences."""
    a, b = flatten(actual), flatten(desired)
    buckets = statistics.mean(a[key]==b[key] for key in PRESENTATION_KEYS)
    moves = difflib.SequenceMatcher(None, [r['role'] for r in actual['paragraph']['moves']],
                                    [r['role'] for r in desired['paragraph']['moves']], autojunk=False).ratio()
    def patterns(structure):
        return [tuple(row[key] for key in ('syntax','condition_position','voice','length')) for row in structure['sentences']]
    syntax = difflib.SequenceMatcher(None, patterns(actual), patterns(desired), autojunk=False).ratio()
    return {'bucket_match':buckets, 'move_sequence_match':moves, 'sentence_sequence_match':syntax,
            'combined':statistics.mean([buckets,moves,syntax])}


def prompt_for(source, condition, source_annotation, target_structure=None):
    if condition not in CONDITIONS: raise ValueError('Unknown oracle condition')
    validate_links(source_annotation['structure'], source_annotation['semantic_ledger'])
    if source_annotation['provenance']['input_sha256'] != sha(source): raise ValueError('Changed source annotation')
    if condition == 'A_plain': return make_prompt(source)
    structure = source_annotation['structure'] if condition == 'B_source_structure' else target_structure
    template = structural_template(structure)
    # Same source evidence in B/C. Never bind target-side IDs to source claims by position.
    claims = [{'claim_id': c['claim_id'], 'source_evidence': c['evidence']['text']}
              for c in source_annotation['semantic_ledger']['claims']]
    constraints = (
        'Use the structural template as a presentation guide, not new factual evidence. '
        'The source-evidence units below must all be preserved; they may contain multiple atomic claims. '
        'Keep their actors, qualifications and semantic relationships fixed. '
        'Move-role and syntax labels are fallible observations, not permission to invent a definition, '
        'contrast, result, citation, technical term or causal connection. '
        'Ignore a template feature if realizing it would add or change a claim. '
        'Sentence slots are not claim identities: combine or split presentation only when meaning is preserved. '
        'Do not output the plan or claim IDs. Return only the rewritten paragraph.'
    )
    return make_prompt(source) + '\n\n' + constraints + '\nSource evidence units:\n' + json.dumps(claims, ensure_ascii=False) + '\nStructural template:\n' + json.dumps(template, sort_keys=True)


def jobs_for(validation):
    if not validation or any(row['split']!='validation' for row in validation): raise ValueError('Oracle requires validation only')
    if len({row['id'] for row in validation})!=len(validation): raise ValueError('Duplicate validation identity')
    jobs = []
    for index, row in enumerate(validation):
        a, b = extract(row['plain_english']), extract(row['completion'])
        # Rotate A/B/C execution order to balance load/order effects; all greedy controls identical.
        order = CONDITIONS[index%3:] + CONDITIONS[:index%3]
        for condition in order:
            prompt = prompt_for(row['plain_english'], condition, a, b['structure'])
            jobs.append({'id': row['id'], 'split': 'validation', 'condition': condition,
                         'prompt': prompt, 'prompt_sha256': sha(prompt), 'source_sha256': sha(row['plain_english']),
                         'oracle': condition == CONDITIONS[2]})
    return jobs


def validate_outputs(jobs, outputs, allow_synthetic=False):
    expected = {(job['id'], job['condition']): job for job in jobs}
    if len(expected)!=len(jobs) or len(outputs)!=len(jobs): raise ValueError('Incomplete oracle outputs')
    seen = set()
    for row in outputs:
        key = (row['id'], row['condition']); job = expected.get(key)
        if key in seen or job is None or row.get('prompt_sha256')!=job['prompt_sha256']:
            raise ValueError('Duplicate, changed or mismatched oracle result')
        if (not isinstance(row.get('output'), str) or not row['output'].strip() or row.get('hit_output_limit') is not False
                or row.get('synthetic') is not allow_synthetic): raise ValueError('Invalid or truncated oracle output')
        seen.add(key)


def generate_jobs(jobs, generator, checkpoint=None):
    """The generator sees a prompt-only job, never paired targets or annotations."""
    results = []
    for job in jobs:
        if sha(job['prompt']) != job['prompt_sha256']: raise ValueError('Changed prompt')
        response = generator(copy.deepcopy(job))
        result = {**response, 'id': job['id'], 'condition': job['condition'],
                  'prompt_sha256': job['prompt_sha256']}
        results.append(result)
        if checkpoint: checkpoint(results)
        if (not isinstance(result.get('output'),str) or not result['output'].strip()
                or result.get('hit_output_limit') is not False):
            raise ValueError('Empty or truncated output; partial record saved, experiment incomplete')
    return results


def score(validation, jobs, outputs, profile, synthetic=False, semantic=None, calibration=None):
    validate_outputs(jobs, outputs, synthetic)
    if any(row['split']!='validation' for row in validation): raise ValueError('Invalid scoring split')
    by_id = {row['id']: row for row in validation}
    records = []
    for output in outputs:
        row = by_id[output['id']]; source, target = row['plain_english'], row['completion']
        generated = extract(output['output'])
        actual = flatten(generated['structure']); target_structure = extract(target)['structure']; oracle = flatten(target_structure)
        source_structure = extract(source)['structure']
        requested = source_structure if output['condition']==CONDITIONS[1] else target_structure if output['condition']==CONDITIONS[2] else None
        oracle_adherence = adherence(generated['structure'],target_structure)
        verification = verify(source, output['output'])
        flags = risk_flags(source, output['output'])
        independent = (semantic or {}).get((row['id'], output['condition']))
        src_tokens, out_tokens = set(tokens(source)), set(tokens(output['output']))
        records.append({'id': row['id'], 'condition': output['condition'],
            'source_similarity': overlap(source, output['output']),
            'exact_copy': ' '.join(source.split())==' '.join(output['output'].split()),
            'near_copy': overlap(source, output['output'])>=.90,
            'semantic_surface_flags': flags, 'bounded_verification': verification['status'],
            'semantic_consistency': independent['status'] if independent else 'unresolved',
            'independent_semantic_assessment_executed': independent is not None,
            'independent_nli': independent,
            'source_token_set_coverage': len(src_tokens & out_tokens)/max(1,len(src_tokens)),
            'output_token_set_support': len(src_tokens & out_tokens)/max(1,len(out_tokens)),
            'oracle_feature_matches': {key: actual[key]==oracle[key] for key in oracle},
            'oracle_structure_match': oracle_adherence['combined'], 'oracle_adherence_components':oracle_adherence,
            'requested_plan_match': None if requested is None else adherence(generated['structure'],requested)['combined'],
            'style_gain_proxy': distance(source, profile)-distance(output['output'], profile),
            'output_input_token_ratio': len(tokens(output['output']))/max(1,len(tokens(source))),
            'input_tokens': output.get('input_tokens'), 'output_tokens': output.get('output_tokens'),
            'generation_seconds': output.get('generation_seconds')})
    aggregate = {}
    for condition in CONDITIONS:
        group = [row for row in records if row['condition']==condition]
        aggregate[condition] = {'rows': len(group),
            **{key: statistics.mean(row[key] for row in group) for key in
               ('source_similarity', 'exact_copy', 'near_copy', 'oracle_structure_match', 'style_gain_proxy', 'output_input_token_ratio')},
            'rows_with_semantic_surface_flags': sum(bool(row['semantic_surface_flags']) for row in group),
            'semantic_consistency_rate': None,
            'requested_plan_match': None if condition==CONDITIONS[0] else statistics.mean(row['requested_plan_match'] for row in group),
            'nli_proxy_consistency_rate': (statistics.mean(row['semantic_consistency']=='proxy_consistent' for row in group)
                                            if calibration and calibration.get('passed') else None),
            'generation_seconds': sum(row['generation_seconds'] or 0 for row in group),
            'input_tokens': sum(row['input_tokens'] or 0 for row in group),
            'output_tokens': sum(row['output_tokens'] or 0 for row in group)}
    lookup = {(row['id'], row['condition']): row for row in records}
    paired = {}
    for comparison in (CONDITIONS[0], CONDITIONS[1]):
        changes = [lookup[(row['id'],CONDITIONS[2])]['oracle_structure_match']-lookup[(row['id'],comparison)]['oracle_structure_match'] for row in validation]
        paired['C_minus_'+comparison] = {'mean_oracle_structure_gain': statistics.mean(changes),
            'improved': sum(value>0 for value in changes), 'worsened': sum(value<0 for value in changes), 'tied': sum(value==0 for value in changes)}
    a, b, c = (aggregate[key] for key in CONDITIONS)
    semantic_ready = not synthetic and bool(calibration and calibration.get('passed')) and all(row['independent_semantic_assessment_executed'] for row in records)
    support = semantic_ready and c['nli_proxy_consistency_rate']>=.9 and c['nli_proxy_consistency_rate']>=max(a['nli_proxy_consistency_rate'],b['nli_proxy_consistency_rate']) and c['near_copy']<=min(a['near_copy'],b['near_copy']) and c['oracle_structure_match']-b['oracle_structure_match']>=.05 and c['style_gain_proxy']>max(a['style_gain_proxy'],b['style_gain_proxy'])
    return records, {'protocol': PROTOCOL, 'execution_kind': 'synthetic_wiring_check' if synthetic else 'model_inference',
        'validation_rows': len(validation), 'conditions': aggregate, 'paired_differences': paired,
        'oracle_is_deployment_result': False, 'test_rows_used': 0, 'training_called': False,
        'oracle_supported_by_automated_proxies': bool(support), 'semantic_claims_verified': False,
        'nli_control_results': calibration,
        'mapping_gate': 'provisionally_supported_by_proxies' if support else 'not_supported_by_current_proxies' if semantic_ready else 'unresolved_semantic_assessment_or_synthetic_run',
        'limitations': 'Adherence and style share surface proxies. Token overlap is not entailment. C uses reference-derived side information; B/C share source evidence while A is the ordinary prompt.'}
