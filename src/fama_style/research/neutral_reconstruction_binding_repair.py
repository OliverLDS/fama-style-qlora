"""Versioned, source-anchored semantic binding repair; no generation or training."""
import argparse
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction import make_handoff, sha, validate_brief, write_json, write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha
from .neutral_reconstruction_review import patch_brief

PROTOCOL = 'neutral_reconstruction_binding_repair_v1'


def apply_repair(record, decision):
    if record['record_id'] != decision['record_id']:
        raise ValueError('repair_record_mismatch')
    reference = record['original_paragraph']
    if sha(reference) != decision['reference_sha256']:
        raise ValueError('reference_binding_changed')
    if sha(json.dumps(record['brief'], sort_keys=True)) != decision['before_brief_sha256']:
        raise ValueError('brief_binding_changed')
    for patch in decision['patches']:
        if (not patch.get('rationale') or not patch.get('source_evidence')
                or patch['source_evidence'] not in reference):
            raise ValueError('source_evidence_or_rationale_missing')
        if not patch['path'] or patch['path'][0] not in ('claims', 'supported_relationships'):
            raise ValueError('repair_outside_semantic_bindings')
    repaired = patch_brief(record['brief'], decision['patches'])
    validate_brief(repaired)
    if [c['claim_id'] for c in repaired['claims']] != [c['claim_id'] for c in record['brief']['claims']]:
        raise ValueError('claim_identity_changed')
    handoff = make_handoff(repaired, reference)
    return {'record_id': record['record_id'], 'brief': repaired, 'handoff': handoff,
            'reference_sha256': sha(reference), 'before_brief_sha256': decision['before_brief_sha256'],
            'after_brief_sha256': sha(json.dumps(repaired, sort_keys=True)),
            'status': 'automated_source_alignment_repair_not_generation_validated',
            'training_ready': False}


def build(packet, decisions, output):
    if output.exists():
        raise ValueError('refuse_overwrite_repair')
    verify_files(packet)
    records = {r['record_id']:r for r in rows(packet/'private_key/reviewed_references.private.jsonl')}
    specification = json.loads(decisions.read_text())
    if specification['protocol'] != PROTOCOL:
        raise ValueError('repair_protocol_mismatch')
    repair_ids = [d['record_id'] for d in specification['decisions']]
    if not repair_ids or len(repair_ids) != len(set(repair_ids)) or not set(repair_ids) <= set(records):
        raise ValueError('repair_coverage')
    repaired = [apply_repair(records[d['record_id']], d) for d in specification['decisions']]
    audit = [{'record_id': d['record_id'], 'original_paragraph':records[d['record_id']]['original_paragraph'],
              'decision':d, 'result':r} for d,r in zip(specification['decisions'],repaired)]
    summary = {'protocol': PROTOCOL, 'records_repaired':len(repaired),
        'binding_issues': ['Confidence negation versus comparative weakening',
                           'Categorical media-authority barrier versus gradual limitation'],
        'fields_changed':sum(len(d['patches']) for d in specification['decisions']),
        'source_references_changed':0, 'raw_responses_changed':0, 'historical_outputs_relabelled':0,
        'derived_briefs_created':len(repaired), 'generation_jobs_prepared':0, 'model_calls':0,
        'gpu_jobs_submitted':0, 'training_pairs_created':0, 'reserved_test_examples_used':0,
        'training_ready':False, 'independent_verification':False,
        'next_gate':'Inference-only transfer check on Gemma 4B before claiming the generator follows this protocol.',
        'limits':'Source-anchored automated repair of reviewed extraction, not new PDF verification or independent semantic gold.'}
    output.mkdir(parents=True)
    write_rows(output/'derived_briefs.private.jsonl',repaired)
    write_rows(output/'repair_audit.private.jsonl',audit)
    write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'source_packet_manifest_sha256':file_sha(packet/'manifest.json'),
        'decision_sha256':file_sha(decisions), 'repair_semantics_version':PROTOCOL,
        'brief_schema_version':'economic_content_brief_v1',
        'scope':'One existing training-side case; no fresh annotation or execution packet.'})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','decisions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.packet,args.decisions,args.output)))


if __name__=='__main__':main()
