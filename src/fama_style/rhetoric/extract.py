"""Independent surface annotation, paired hypotheses, and constrained operations."""
import collections
import json
import re

from fama_style.research.claims import ledger, sha, tokens, sentences, conditional, variants, verify, risk_flags
from fama_style.research.signal import overlap
from .schema import VERSION, EXTRACTOR, validate_links, validate

PROTECTED_FEATURES = {'hedging', 'citation'}


def count_bin(n):
    return 'one' if n == 1 else 'two' if n == 2 else 'three_plus'


def rate_bin(n, total):
    return 'none' if not n else 'low' if n/max(1, total) <= .025 else 'high'


def role(text):
    """Cue observations, never assertions that a relation is semantically valid."""
    low = text.lower()
    if re.search(r'\b(?:is defined as|we define|denotes)\b', low): return 'definition'
    if re.match(r'(?:however|nevertheless|in contrast|but)\b', low): return 'contrast'
    if re.match(r'(?:therefore|thus|hence|in sum)\b', low): return 'conclusion'
    if low.startswith(('if ', 'unless ', 'when ')): return 'condition'
    if re.search(r'\b(?:we find|results show|table \d|we estimate|we test)\b', low): return 'reporting'
    return 'assertion'


def extract(text):
    if not isinstance(text, str) or not text.strip(): raise ValueError('Empty paragraph')
    claims = ledger(text)
    parts = sentences(text)
    units, moves = [], []
    for index, part in enumerate(parts, 1):
        subordinated = bool(re.search(r'\b(if|unless|when|although|because|which|that|while|whereas)\b', part, re.I))
        coordinated = bool(re.search(r';|,\s+(?:and|but|or)\b', part, re.I))
        syntax = 'mixed' if subordinated and coordinated else 'subordinated' if subordinated else 'coordinated' if coordinated else 'simple'
        position = 'none'
        if re.search(r'\b(if|unless|when|provided that)\b', part, re.I):
            position = 'front' if re.match(r'(if|unless|when|provided that)\b', part, re.I) else 'end' if conditional(part) else 'embedded'
        n = len(tokens(part)); unit = f's{index}'; refs = [f'c{index}']
        units.append({'unit': unit, 'claims': refs, 'syntax': syntax,
                      'condition_position': position, 'voice': 'present' if re.search(r'\b(we|our|us)\b', part, re.I) else 'absent',
                      'length': 'short' if n <= 16 else 'medium' if n <= 32 else 'long'})
        moves.append({'unit': unit, 'claims': refs, 'role': role(part)})
    roles = {move['role'] for move in moves}
    specials = roles & {'contrast', 'condition', 'reporting'}
    organization = 'mixed' if len(specials)>1 else 'contrastive' if 'contrast' in roles else 'conditional' if 'condition' in roles else 'reporting' if 'reporting' in roles else 'single' if len(parts)==1 else 'sequence'
    words = tokens(text); n = len(words)
    structure = {'version': VERSION,
        'paragraph': {'organization': organization, 'sentence_count': count_bin(len(parts)), 'moves': moves},
        'sentences': units,
        'language': {'author_voice': 'present' if any(unit['voice']=='present' for unit in units) else 'absent',
            'semicolon': rate_bin(text.count(';'), n), 'parenthesis': rate_bin(text.count('('), n),
            'hedging': rate_bin(len(re.findall(r'\b(may|might|could|possibly|perhaps)\b', text, re.I)), n),
            'citation': rate_bin(len(re.findall(r'\b(?:18|19|20)\d{2}\b', text)), n),
            'nominalization': rate_bin(sum(bool(re.search(r'(tion|sion|ment|ance|ence|ity)$', word)) for word in words), n)}}
    validate_links(structure, claims)
    return {'structure': structure, 'semantic_ledger': claims,
        'provenance': {'specification': VERSION, 'extractor': EXTRACTOR, 'input_sha256': sha(text),
                       'annotation_method': 'deterministic_surface_rules', 'independent_of_paired_text': True},
        'checks': {'claim_links_valid': True, 'semantic_parse_verified': False,
                   'atomic_claim_coverage': 'unresolved', 'unsupported_relation_creation_allowed': False}}


def flatten(structure):
    validate(structure)
    sentence = structure['sentences']
    def majority(key):
        counts = collections.Counter(row[key] for row in sentence)
        return sorted(counts, key=lambda item:(-counts[item], item))[0]
    return {'organization': structure['paragraph']['organization'], 'sentence_count': structure['paragraph']['sentence_count'],
            'syntax': majority('syntax'), 'condition_position': majority('condition_position'),
            'sentence_length': majority('length'), **structure['language']}


def paired(source, target):
    """Extract separately; alignment is an explicitly unverified lexical hypothesis."""
    a, b = extract(source), extract(target)
    hypotheses = []
    for target_claim in b['semantic_ledger']['claims']:
        ranked = sorted(((overlap(claim['evidence']['text'], target_claim['evidence']['text']), claim['claim_id'])
                         for claim in a['semantic_ledger']['claims']), reverse=True)
        score, claim_id = ranked[0]
        hypotheses.append({'target_claim': target_claim['claim_id'], 'source_claim': claim_id if score >= .35 else None,
                           'similarity': score, 'status': 'lexical_hypothesis_not_semantically_validated'})
    fa, fb = flatten(a['structure']), flatten(b['structure'])
    refs = [claim['claim_id'] for claim in a['semantic_ledger']['claims']]
    changes = [{'operation': 'preserve_source_feature' if key in PROTECTED_FEATURES else 'set_template_feature',
                'feature': key, 'from': fa[key], 'to': fa[key] if key in PROTECTED_FEATURES else fb[key],
                'observed_target': fb[key], 'status': 'blocked_semantic_sensitive' if key in PROTECTED_FEATURES else 'structural_hypothesis',
                'source_claim_refs': refs, 'scope': 'planning_only', 'introduces_semantic_relation': False}
               for key in fa if fa[key] != fb[key]]
    # Programs are replayable surface controls, not a claim that they realize the target.
    programs = [{'operation': item['operation'], 'source_claim_refs': refs,
                 'output_sha256': sha(item['text']), 'verification': verify(source, item['text']),
                 'target_feature_matches': sum(value == flatten(extract(item['text'])['structure'])[key] for key,value in fb.items())}
                for item in variants(source) if item['operation']!='identity']
    return {'source': a, 'target': b, 'claim_alignment': hypotheses, 'template_changes': changes,
            'executable_surface_controls': programs,
            'consistency': {'source_target_surface_flags': risk_flags(source, target),
                            'semantic_alignment_verified': False, 'claim_reordering_allowed': False}}


def execute_operation(text, operation, claim_refs, expected_source_sha):
    """An operation can touch only this source's known claims and bounded grammar."""
    if sha(text) != expected_source_sha: raise ValueError('Changed operation source')
    known = {c['claim_id'] for c in ledger(text)['claims']}
    if not claim_refs or set(claim_refs) != known: raise ValueError('Invalid operation claim scope')
    candidate = next((item for item in variants(text) if item['operation']==operation), None)
    if candidate is None or verify(text, candidate['text'])['status']!='surface_preserved':
        raise ValueError('Unsupported semantic or structural operation')
    return candidate['text']


def inspect_legacy(rows):
    shapes = collections.Counter(); labels = collections.Counter()
    for row in rows:
        value = row.get('rhetorical_taxonomy')
        shape = 'missing' if value is None else ','.join(sorted(value)) if isinstance(value, dict) else 'unexpected'
        shapes[shape] += 1
        if isinstance(value, dict):
            # Inventory only pre-existing finite objectives, not private descriptions.
            top = value.get('level_1', value.get('R1', {}))
            objective = top.get('discourse_objective') if isinstance(top, dict) else None
            if isinstance(objective, str) and re.fullmatch(r'[A-Z_]{1,32}', objective): labels[objective] += 1
    return {'rows': len(rows), 'formats': dict(shapes), 'objective_counts': dict(labels),
            'reused_as_source_labels': False, 'migration': 'preserved unchanged; discourse objective/moves/relations do not match new three-level axes'}
