"""Import broader author annotations, with bounded repairs and conservative holds.

CPU-only. Raw files, original prose, ledgers and prior releases are immutable.
Candidate status is not independent semantic verification or training approval.
"""
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from .human_prose_annotations import expand_catalog
from .human_prose_review import audit_record, normalized, schema_errors

PROTOCOL = 'author_pair_review_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def repair_schema_direction(row):
    """Swap containers only when exact full-text coverage proves reversal.

    No ledger, transformation, evidence quote or text is inferred or rewritten.
    Identical texts and partial/ambiguous matches cannot establish reversal.
    """
    result = copy.deepcopy(row)
    source, target = row['plain_english'], row['original_paragraph']
    if not source or normalized(source) == normalized(target):
        return result, []
    def covered(side):
        return normalized(' '.join(s['exact_text'] for s in row[side]['sentences'])) if row[side] else None
    if covered('source_schema') == normalized(target) and covered('target_schema') == normalized(source):
        before = object_sha({'source_schema':result['source_schema'], 'target_schema':result['target_schema']})
        result['source_schema'], result['target_schema'] = result['target_schema'], result['source_schema']
        after = object_sha({'source_schema':result['source_schema'], 'target_schema':result['target_schema']})
        return result, [{'operation':'swap_proven_reversed_schema_containers',
                         'before_sha256':before, 'after_sha256':after,
                         'basis':'Both complete ordered sentence lists exactly match the opposite text.'}]
    return result, []


def pair_holds(row, audit):
    reasons = []
    if row['source_quality']['status'] != 'usable': reasons.append('source_quality_not_usable')
    if row['source_quality']['needs_external_context']: reasons.append('missing_external_context')
    if row['needs_review']: reasons.append('annotator_unresolved_review')
    if row['rewrite_status'] == 'pending_review' or not row['plain_english']: reasons.append('rewrite_pending_or_absent')
    if audit['errors']: reasons.append('mechanical_annotation_errors')
    if any(c['alignment_status'] != 'preserved' for c in row['semantic_ledger']['claims']):
        reasons.append('claim_alignment_not_preserved')
    if any(c['alignment_status'] != 'preserved' for c in row['semantic_ledger']['relations']):
        reasons.append('relation_alignment_not_preserved')
    if row['semantic_ledger']['unmatched_source_claims']: reasons.append('unsupported_rewrite_claims')
    if any(w['code'] == 'numeric_surface_change_needs_scope_check' for w in audit['warnings']):
        reasons.append('numeric_scope_check_pending')
    return reasons


def structural_holds(audit):
    return sorted(set([e['code'] for e in audit['program_errors']] + [
        w['code'] for w in audit['warnings'] if w['code'] in (
            'observation_claim_outside_bound_sentences', 'uncertain_transformation',
            'taxonomy_unresolved', 'taxonomy_not_covered')]))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in rows))


def review(release, output, dataset_output, reported_model='GPT-6', reported_effort='High'):
    release = release.resolve()
    if output.exists() or dataset_output.exists():
        raise ValueError('fresh_versioned_output_required')
    provenance = {'user_reported_model':reported_model, 'user_observed_thinking_effort':reported_effort,
                  'source':'User report, 2026-11-03; all batches observed in High thinking mode.',
                  'raw_null_values_preserved':True, 'actual_backend_effort_verified':False,
                  'conversation_arrangement':'not newly specified', 'cross_chat_independence_assumed':False}
    output.mkdir(parents=True)
    records, coverage, repairs, pairs = [], [], [], []
    raw_provenance = Counter(); inputs = {}; seen_paragraphs = set()
    inventory_path = release.parents[2]/'docs/author_inventory_v3/inventory.json'
    inventory = {r['paper_id']:r for r in json.loads(inventory_path.read_text())['papers']}
    inputs[str(inventory_path)] = sha(inventory_path)
    for package_name in ('existing_collection', 'new_collection'):
        package = release/package_name
        manifest = json.loads((package/'manifest.json').read_text())
        for relative, expected in manifest['output_sha256'].items():
            if relative == 'session_notes.json': continue  # Explicitly editable operator notes.
            path = (package/relative).resolve()
            if package not in path.parents or sha(path) != expected:
                raise ValueError('package_integrity_failure:'+relative)
        inputs[str(package/'manifest.json')] = sha(package/'manifest.json')
        catalog = {r['canonical_id']:r for r in expand_catalog(json.loads(
            (package/'common/taxonomy_catalog.compact.json').read_text()))}
        schema = json.loads((package/'common/annotation_output.schema.json').read_text())
        keys = {r['paragraph_id']:r for r in json.loads((package/'selection.private.json').read_text())}
        roles = {r['paper_id']:r['role'] for r in json.loads((package/'PAPER_ROLES.json').read_text())['papers']}
        for batch in json.loads((package/'BATCH_INDEX.json').read_text()):
            if batch['role'] not in ('training','development') or roles[batch['paper_id']] != batch['role']:
                raise ValueError('invalid_batch_role')
            path = (package/batch['save_response']).resolve()
            response_root = release/'responses'/package_name/batch['role']
            if path.parent != response_root or path.suffix != '.jsonl':
                raise ValueError('response_path_outside_declared_role')
            raw = path.read_bytes(); inputs[str(path)] = sha(path)
            snapshot = output/'raw'/package_name/batch['role']; snapshot.mkdir(parents=True, exist_ok=True)
            (snapshot/path.name).write_bytes(raw)
            report = path.with_name(path.stem.replace('responses_','coverage_')+'.md')
            if report.exists():
                shutil.copyfile(report, snapshot/report.name); inputs[str(report)] = sha(report)
            expected = {r['paragraph_id']:r['text'] for line in (package/batch['input']).read_text().splitlines()
                        if (r := json.loads(line))}
            rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
            ids = [r.get('paragraph_id') for r in rows if isinstance(r,dict)]
            if len(ids) != len(expected) or set(ids) != set(expected) or seen_paragraphs & set(ids):
                raise ValueError('incomplete_duplicate_or_unexpected_response_ids')
            seen_paragraphs.update(ids)
            batch_coverage = {'package':package_name, 'batch_id':batch['batch_id'], 'paper_id':batch['paper_id'],
                              'role':batch['role'], 'expected':len(expected), 'received':len(rows),
                              'schema_valid':0, 'originals_exact':True}
            for row in rows:
                pid = row['paragraph_id']; key = keys[pid]
                paper = inventory[batch['paper_id']]
                if key['paper_id'] != batch['paper_id'] or key['role'] != batch['role']:
                    raise ValueError('selection_role_mismatch')
                if row['original_paragraph'] != expected[pid] or row['batch_id'] != batch['batch_id']:
                    raise ValueError('original_or_batch_changed')
                if hashlib.sha256(row['original_paragraph'].encode()).hexdigest() != key['original_sha256']:
                    raise ValueError('original_hash_mismatch')
                validation = schema_errors(row, schema)
                raw_provenance[(row.get('provenance',{}).get('visible_model_label'),
                                row.get('provenance',{}).get('thinking_setting'))] += 1
                base = {'package':package_name, 'batch_id':batch['batch_id'], 'paper_id':batch['paper_id'],
                        'style_group':batch['style_group'], 'role':batch['role'], 'paragraph_id':pid,
                        'source_file_sha256':inputs[str(path)], 'original_sha256':key['original_sha256'],
                        'schema_errors':validation}
                if validation:
                    records.append(dict(base, annotation=row, raw_audit=None, audit=None,
                                        pair_hold_reasons=['schema_invalid'], structural_hold_reasons=['schema_invalid']))
                    continue
                batch_coverage['schema_valid'] += 1
                raw_audit = audit_record(row,catalog)
                effective, patches = repair_schema_direction(row)
                audited = audit_record(effective,catalog)
                for patch in patches:
                    repairs.append(dict(patch,package=package_name,paragraph_id=pid,
                        original_sha256_unchanged=key['original_sha256'],
                        source_text_sha256_unchanged=hashlib.sha256(row['plain_english'].encode()).hexdigest(),
                        ledger_sha256_unchanged=object_sha(row['semantic_ledger']),
                        transformations_sha256_unchanged=object_sha(row['transformations'])))
                hold = pair_holds(effective,audited)
                structural = structural_holds(audited)
                records.append(dict(base, annotation=effective, raw_audit=raw_audit, audit=audited,
                                    pair_hold_reasons=hold, structural_hold_reasons=structural))
                if not hold:
                    pairs.append({'example_id':'APE_'+pid, 'paragraph_id':pid, 'paper_id':batch['paper_id'],
                                  'style_group':batch['style_group'], 'split':batch['role'],
                                  'authors':paper['authors'], 'title':paper['title'],
                                  'version_year':paper['version_year'], 'pdf_sha256':paper['pdf_sha256'],
                                  'source':effective['plain_english'], 'target':effective['original_paragraph'],
                                  'source_sha256':hashlib.sha256(effective['plain_english'].encode()).hexdigest(),
                                  'target_sha256':key['original_sha256'], 'source_provenance':key,
                                  'annotation_file_sha256':inputs[str(path)], 'structural_annotation_candidate':not structural,
                                  'training_ready':False, 'independent_semantic_verification':False,
                                  'status':'provisional_selfchecked_pair_mechanical_audit_only'})
            coverage.append(batch_coverage)
    def counter(field, code_field):
        return dict(Counter(e['code'] for r in records if r[field] for e in r[field][code_field]))
    groups = []
    for group in sorted({r['style_group'] for r in records}):
        items = [r for r in records if r['style_group']==group]
        eligible = [p for p in pairs if p['style_group']==group]
        groups.append({'style_group':group,'received':len(items),
                       'provisional_training':sum(p['split']=='training' for p in eligible),
                       'provisional_development':sum(p['split']=='development' for p in eligible),
                       'provisional_papers':len({p['paper_id'] for p in eligible})})
    ratios = [r['audit']['metrics']['lexical_overlap_sequence_ratio'] for r in records
              if not r['pair_hold_reasons'] and r['audit']['metrics']]
    summary = {'protocol':PROTOCOL,'status':'responses_imported_provisional_pairs_not_training_approved',
        'batches':len(coverage),'expected_rows':sum(r['expected'] for r in coverage),'received_rows':len(records),
        'schema_valid_rows':sum(r['schema_valid'] for r in coverage),'originals_unchanged':True,
        'raw_mechanical_error_rows':sum(bool(r['raw_audit'] and r['raw_audit']['errors']) for r in records),
        'effective_mechanical_error_rows':sum(bool(r['audit'] and r['audit']['errors']) for r in records),
        'raw_mechanical_errors':counter('raw_audit','errors'), 'effective_mechanical_errors':counter('audit','errors'),
        'effective_program_error_rows':sum(bool(r['audit'] and r['audit']['program_errors']) for r in records),
        'effective_program_errors':counter('audit','program_errors'),
        'repair_rows':len(repairs), 'repair_type':'schema-container swap proven by opposite full-text coverage',
        'source_quality':dict(Counter(r['annotation']['source_quality']['status'] for r in records)),
        'needs_external_context':sum(r['annotation']['source_quality']['needs_external_context'] for r in records),
        'annotator_needs_review':sum(r['annotation']['needs_review'] for r in records),
        'rewrite_status':dict(Counter(r['annotation']['rewrite_status'] for r in records)),
        'provisional_pairs':len(pairs),'provisional_by_split':dict(Counter(p['split'] for p in pairs)),
        'provisional_by_group':groups,'quarantined_rows':sum(bool(r['pair_hold_reasons']) for r in records),
        'pair_hold_reasons':dict(Counter(reason for r in records for reason in r['pair_hold_reasons'])),
        'structural_candidates':sum(p['structural_annotation_candidate'] for p in pairs),
        'candidate_overlap':{'median_sequence_ratio':statistics.median(ratios) if ratios else None,
                             'min_sequence_ratio':min(ratios) if ratios else None,
                             'max_sequence_ratio':max(ratios) if ratios else None},
        'raw_provenance_counts':[{'model':k[0],'thinking_effort':k[1],'rows':v} for k,v in raw_provenance.items()],
        'session_provenance':provenance,'training_ready_rows':0,'independent_semantic_verification':False,
        'reserved_text_bodies_read':0,'GPU_jobs':0,'model_calls':0,'raw_responses_changed':False}
    dataset_output.mkdir(parents=True)
    write_rows(output/'audits.private.jsonl',records)
    write_rows(output/'repairs.private.jsonl',repairs)
    write_json(output/'coverage.json',coverage)
    write_json(output/'session_provenance.json',provenance)
    write_json(output/'summary.json',summary)
    for split in ('training','development'):
        write_rows(dataset_output/(split+'.provisional.private.jsonl'),[p for p in pairs if p['split']==split])
    write_rows(dataset_output/'quarantine.private.jsonl',[{'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],
        'style_group':r['style_group'],'role':r['role'],'reasons':r['pair_hold_reasons'],
        'annotation':r['annotation'],'audit':r['audit']} for r in records if r['pair_hold_reasons']])
    write_json(dataset_output/'summary.json',summary)
    for path,expected in inputs.items():
        if sha(Path(path)) != expected: raise ValueError('input_changed_during_review')
    files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'input_sha256':inputs,
        'code_sha256':sha(Path(__file__)),'files':files,
        'dependency_sha256':{name:sha(Path(__file__).parent/name) for name in (
            'human_prose_review.py','human_prose_annotations.py')},
        'dataset_files':{p.name:sha(p) for p in sorted(dataset_output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.release,args.output,args.dataset_output),indent=2))


if __name__=='__main__': main()
