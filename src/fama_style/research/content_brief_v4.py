"""Compact unordered content briefs; versioned independently of earlier contracts."""
import copy
import re
from .content_brief_v3 import PURPOSES, RELATION_SPEC, validate_brief_v3


def validate_brief_v4(row):
    if not isinstance(row, dict) or set(row) != {'id', 'target_sha256', 'brief', 'status', 'notes'}:
        raise ValueError('brief_fields')
    b = row['brief']
    if not isinstance(b, dict) or set(b) != {'topic', 'purpose', 'key_points', 'anchors', 'relationships'}:
        raise ValueError('v4_brief_fields')
    if not isinstance(b['topic'], str) or not b['topic'].strip():
        raise ValueError('brief_topic')
    if not isinstance(b['purpose'], dict) or set(b['purpose']) != {'primary', 'secondary'}:
        raise ValueError('v4_purpose_fields')
    if not isinstance(b['anchors'], dict):
        raise ValueError('v4_anchors')
    for key, value in b['anchors'].items():
        if not re.fullmatch(r'A[1-9][0-9]*', key) or not isinstance(value, str) or not value.strip():
            raise ValueError('anchor_entry')
        if re.search(r'@A[0-9]+\b', value):
            raise ValueError('nested_anchor_reference')
    if not isinstance(b['key_points'], list) or any(not isinstance(x, str) for x in b['key_points']):
        raise ValueError('brief_points')
    if not isinstance(b['relationships'], list):
        raise ValueError('brief_relations')
    texts = [b['topic']] + b['key_points']
    for relation in b['relationships']:
        if not isinstance(relation, dict) or not isinstance(relation.get('qualification'), str):
            raise ValueError('brief_relation')
        texts.append(relation['qualification'])
    for text in texts:
        for ref in re.findall(r'@A[0-9]+\b', text):
            if ref[1:] not in b['anchors']:
                raise ValueError('dangling_anchor')
    # Reuse established field/role/point-reference validation on a disposable
    # shape adapter. No fabricated takeaway is stored or sent to generation.
    checked = copy.deepcopy(row)
    checked['brief'] = dict(task=b['topic'], purpose=dict(b['purpose'], reader_takeaway='validation adapter'),
        key_points=b['key_points'], anchors=list(b['anchors'].values()), relationships=b['relationships'])
    validate_brief_v3(checked)
    return row


def canonicalize_note_order(row):
    """Remove incidental list order while preserving directed point bindings."""
    validate_brief_v4(row)
    result = copy.deepcopy(row)
    notes = result['brief']['key_points']
    order = sorted(range(len(notes)), key=lambda i: (notes[i].casefold(), notes[i]))
    remap = {f'K{old + 1}': f'K{new + 1}' for new, old in enumerate(order)}
    result['brief']['key_points'] = [notes[i] for i in order]
    for edge in result['brief']['relationships']:
        edge['from_point'] = remap[edge['from_point']]
        edge['to_point'] = remap[edge['to_point']]
    result['brief']['relationships'].sort(key=lambda x: (x['type'], x['from_point'], x['to_point'], x['qualification']))
    validate_brief_v4(result)
    return result


def reconstruction_conditions(row):
    """Share every factual note/anchor; vary explicit purpose and relationships."""
    row = canonicalize_note_order(row)
    b = row['brief']
    core = dict(id=row['id'], target_sha256=row['target_sha256'], brief_status=row['status'],
        brief_notes=copy.deepcopy(row['notes']), brief={key: copy.deepcopy(b[key]) for key in ('topic', 'key_points', 'anchors')})
    guided = copy.deepcopy(core)
    guided['brief'] = copy.deepcopy(b)
    return dict(content_only=core, purpose_and_relations=guided)
