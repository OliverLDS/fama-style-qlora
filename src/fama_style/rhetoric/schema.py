"""Closed, content-free structural vocabulary. Claims live in a separate ledger."""
import re

VERSION = 'rhetoric_v1'
EXTRACTOR = 'surface_rules_v1'
ROLES = ['assertion', 'definition', 'reporting', 'contrast', 'condition', 'conclusion', 'unknown']
ORGANIZATIONS = ['single', 'sequence', 'conditional', 'contrastive', 'reporting', 'mixed']
BINS = ['none', 'low', 'high']
COUNTS = ['one', 'two', 'three_plus']
LENGTHS = ['short', 'medium', 'long']
SYNTAX = ['simple', 'coordinated', 'subordinated', 'mixed']
POSITIONS = ['none', 'front', 'end', 'embedded', 'mixed']
VOICES = ['absent', 'present']


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def enum(values):
    return {'type': 'string', 'enum': values}


def array(item):
    return {'type': 'array', 'items': item, 'minItems': 1}


UNIT = {'type': 'string', 'pattern': r'^s[1-9][0-9]*$'}
CLAIMS = array({'type': 'string', 'pattern': r'^c[1-9][0-9]*$'})
STRUCTURE_SCHEMA = obj({
    'version': enum([VERSION]),
    'paragraph': obj({'organization': enum(ORGANIZATIONS), 'sentence_count': enum(COUNTS),
        'moves': array(obj({'unit': UNIT, 'role': enum(ROLES), 'claims': CLAIMS}))}),
    'sentences': array(obj({'unit': UNIT, 'claims': CLAIMS, 'syntax': enum(SYNTAX),
        'condition_position': enum(POSITIONS), 'voice': enum(VOICES), 'length': enum(LENGTHS)})),
    'language': obj({'author_voice': enum(VOICES), 'semicolon': enum(BINS),
        'parenthesis': enum(BINS), 'hedging': enum(BINS), 'citation': enum(BINS),
        'nominalization': enum(BINS)}),
})
JSON_SCHEMA = {'$schema': 'https://json-schema.org/draft/2020-12/schema',
               '$id': 'urn:fama-style:rhetoric:v1', **STRUCTURE_SCHEMA}


def validate(value, schema=STRUCTURE_SCHEMA):
    """Validate this closed subset without requiring a local JSON Schema dependency."""
    kind = schema['type']
    if kind == 'object':
        if not isinstance(value, dict) or set(value) != set(schema['properties']):
            raise ValueError('Unexpected or missing structural fields')
        for key, child in schema['properties'].items():
            validate(value[key], child)
    elif kind == 'array':
        if not isinstance(value, list) or len(value) < schema['minItems']:
            raise ValueError('Invalid structural array')
        for item in value:
            validate(item, schema['items'])
    elif kind == 'string':
        if not isinstance(value, str):
            raise ValueError('Invalid structural label')
        if 'enum' in schema and value not in schema['enum']:
            raise ValueError('Unknown structural label')
        if 'pattern' in schema and not re.fullmatch(schema['pattern'], value):
            raise ValueError('Invalid claim/unit identity')
    return value


def validate_links(structure, ledger):
    validate(structure)
    ids = {claim['claim_id'] for claim in ledger['claims']}
    units = [row['unit'] for row in structure['sentences']]
    if len(set(units)) != len(units):
        raise ValueError('Duplicate structural unit')
    if [row['unit'] for row in structure['paragraph']['moves']] != units:
        raise ValueError('Paragraph moves and sentence units disagree')
    if set(ref for row in structure['sentences'] for ref in row['claims']) != ids:
        raise ValueError('Uncovered or invented claim IDs')
    for move, sentence in zip(structure['paragraph']['moves'], structure['sentences']):
        if move['claims'] != sentence['claims']:
            raise ValueError('Move claim binding disagrees with sentence binding')
    return True
