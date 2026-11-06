"""Small content-brief contract for purpose-aware reconstruction; no model imports."""
import re

PURPOSES = (
    'introduce_topic', 'define_concept', 'describe_phenomenon', 'explain_mechanism',
    'explain_difference', 'compare_accounts', 'report_result', 'describe_method',
    'introduce_data', 'provide_background', 'illustrate_case', 'state_problem',
    'draw_implication', 'qualify_claim', 'evaluate_argument', 'recommend_action',
    'outline_paper', 'mixed_or_uncertain',
)
RELATIONS = ('cause', 'association', 'contrast', 'condition', 'temporal_sequence',
             'evidence_for', 'definition_part', 'qualification', 'alternative')


def validate_brief_v2(row):
    if set(row) != {'id', 'target_sha256', 'brief', 'status', 'notes'}:
        raise ValueError('brief_fields')
    if not isinstance(row['id'], str) or not row['id'] or not isinstance(row['target_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['target_sha256']):
        raise ValueError('brief_identity')
    b = row['brief']
    if not isinstance(b, dict) or set(b) != {'task', 'purpose', 'key_points', 'anchors', 'relationships'}:
        raise ValueError('brief_v2_fields')
    if not isinstance(b['task'], str) or not b['task'].strip():
        raise ValueError('brief_task')
    for key in ('key_points', 'anchors'):
        if not isinstance(b[key], list) or any(not isinstance(x, str) or not x.strip() for x in b[key]):
            raise ValueError('brief_content')
    if not b['key_points']:
        raise ValueError('empty_points')
    p = b['purpose']
    if not isinstance(p, dict) or set(p) != {'primary', 'secondary', 'reader_takeaway'}:
        raise ValueError('brief_purpose')
    if p['primary'] not in PURPOSES or not isinstance(p['secondary'], list) or len(p['secondary']) > 2 or any(x not in PURPOSES for x in p['secondary']):
        raise ValueError('purpose_labels')
    if len(set(p['secondary'])) != len(p['secondary']) or p['primary'] in p['secondary']:
        raise ValueError('duplicate_purpose')
    if not isinstance(p['reader_takeaway'], str) or not p['reader_takeaway'].strip():
        raise ValueError('reader_takeaway')
    if not isinstance(b['relationships'], list):
        raise ValueError('brief_relations')
    for relation in b['relationships']:
        if not isinstance(relation, dict) or set(relation) != {'from_point', 'to_point', 'type', 'qualification'} or relation['type'] not in RELATIONS or not isinstance(relation['qualification'], str):
            raise ValueError('relation_fields')
        for key in ('from_point', 'to_point'):
            value = relation[key]
            if not isinstance(value, str) or not re.fullmatch(r'K[1-9][0-9]*', value) or int(value[1:]) > len(b['key_points']):
                raise ValueError('relation_point')
        if relation['from_point'] == relation['to_point']:
            raise ValueError('self_relation')
    if row['status'] not in {'ready', 'source_problem'} or not isinstance(row['notes'], list) or any(not isinstance(x, str) for x in row['notes']):
        raise ValueError('brief_status')
    if row['status'] == 'source_problem' and not row['notes']:
        raise ValueError('unexplained_source_problem')
    return row
