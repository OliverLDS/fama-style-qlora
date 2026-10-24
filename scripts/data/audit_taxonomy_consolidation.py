"""Check a proposed vocabulary against its exact discovery inventory; no models."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check(taxonomy,inventory,occurrences):
    errors=[]
    def finding(code,**extra):errors.append({'code':code,**extra})
    items=taxonomy.get('canonical_items',[]);mappings=taxonomy.get('inventory_mapping',[])
    inv={r['inventory_id']:r for r in inventory};canonical={r['canonical_id']:r for r in items};mapping={r['inventory_id']:r for r in mappings}
    if len(inv)!=len(inventory):finding('duplicate_inventory_id')
    if len(canonical)!=len(items):finding('duplicate_canonical_id')
    if len(mapping)!=len(mappings):finding('duplicate_mapping_id')
    if set(mapping)!=set(inv):finding('mapping_coverage_mismatch',missing=sorted(set(inv)-set(mapping)),unknown=sorted(set(mapping)-set(inv)))
    if taxonomy.get('status')!='proposed_not_frozen':finding('unexpected_taxonomy_status')
    fields={'canonical_id','level','dimension','name','definition','inclusion_rules','exclusion_rules','allowed_values',
        'applicability','evidence_requirements','control_status','preconditions','prohibited_effects','supporting_inventory_ids','related_item_ids','provisional_reason'}
    statuses={'mapped','split_requires_context','unresolved'}
    reverse={key:set() for key in canonical};safe_primary={key:set() for key in canonical};safe_all={key:set() for key in canonical}
    for row in mappings:
        ident=row['inventory_id'];status=row.get('status');refs=row.get('canonical_ids',[])
        if status not in statuses:finding('invalid_mapping_status',inventory_id=ident)
        if len(refs)!=len(set(refs)):finding('duplicate_mapping_candidate',inventory_id=ident)
        if status=='mapped' and len(refs)!=1:finding('mapped_not_one_to_one',inventory_id=ident)
        if status=='split_requires_context' and (len(refs)<2 or not row.get('disambiguation_rule')):
            finding('split_missing_alternatives_or_rule',inventory_id=ident)
        for ref in refs:
            if ref not in canonical:finding('unknown_canonical_reference',inventory_id=ident,canonical_id=ref);continue
            reverse[ref].add(ident)
            if ident in inv:
                if any(inv[ident][key]!=canonical[ref].get(key) for key in ('level','dimension')):
                    finding('cross_level_or_dimension_mapping',inventory_id=ident,canonical_id=ref)
                if status=='mapped':
                    safe_all[ref].add(ident)
                    if inv[ident].get('primary_context_counts'):safe_primary[ref].add(ident)
    for row in items:
        ident=row['canonical_id']
        if not fields<=set(row):finding('missing_canonical_fields',canonical_id=ident)
        if row.get('level') not in ('paragraph','sentence','language'):finding('invalid_canonical_level',canonical_id=ident)
        if row.get('control_status') not in ('descriptive','candidate_presentation_control','semantic_sensitive_protected'):
            finding('invalid_control_status',canonical_id=ident)
        for field in ('inclusion_rules','exclusion_rules','allowed_values','evidence_requirements','preconditions','prohibited_effects','supporting_inventory_ids','related_item_ids'):
            if not isinstance(row.get(field),list):finding('invalid_canonical_array',canonical_id=ident,field=field)
        support=row.get('supporting_inventory_ids',[])
        if len(support)!=len(set(support)):finding('duplicate_support_reference',canonical_id=ident)
        if set(support)!=reverse[ident]:finding('support_mapping_inverse_mismatch',canonical_id=ident)
        if any(ref not in inv for ref in support):finding('unknown_support_id',canonical_id=ident)
        if any(ref not in canonical for ref in row.get('related_item_ids',[])):finding('unknown_related_id',canonical_id=ident)
        if row.get('control_status')=='candidate_presentation_control' and (not row.get('preconditions') or not row.get('prohibited_effects')):
            finding('presentation_control_missing_guards',canonical_id=ident)
    if set(row['inventory_id'] for row in occurrences)!=set(inv):finding('occurrence_inventory_coverage_mismatch')
    observed=collections.Counter(row['inventory_id'] for row in occurrences)
    if any(observed[key]!=row['occurrence_count'] for key,row in inv.items()):finding('occurrence_count_mismatch')
    decisions=taxonomy.get('merge_and_split_decisions',[])
    for index,decision in enumerate(decisions):
        for ref in decision.get('affected_inventory_ids',[]):
            if ref not in inv:finding('unknown_decision_inventory_id',decision=index)
    for index,issue in enumerate(taxonomy.get('unresolved_items',[])):
        if any(ref not in inv for ref in issue.get('affected_inventory_ids',[])):finding('unknown_unresolved_inventory_id',issue=index)
    status_counts=dict(collections.Counter(row['status'] for row in mappings))
    occurrence_counts=dict(collections.Counter({status:sum(inv[row['inventory_id']]['occurrence_count'] for row in mappings if row['inventory_id'] in inv and row['status']==status) for status in sorted(statuses)}))
    level_counts=dict(collections.Counter(row['level'] for row in items));control_counts=dict(collections.Counter(row['control_status'] for row in items))
    reported=taxonomy.get('consistency_checks',{})
    for key,value in {'input_inventory_rows_read':len(inventory),'input_unique_inventory_ids':len(inv),
        'mapped_rows':status_counts.get('mapped',0),'split_requires_context_rows':status_counts.get('split_requires_context',0),
        'unresolved_rows':status_counts.get('unresolved',0),'inventory_mapping_entries':len(mappings),
        'source_label_occurrences_reported':len(occurrences),'canonical_item_counts':level_counts,
        'canonical_control_status_counts':control_counts}.items():
        if key in reported and reported[key]!=value:finding('reported_count_mismatch',field=key)
    summary={'canonical_categories':len(items),'inventory_items':len(inventory),'label_occurrences':len(occurrences),
        'mapping_status_counts':status_counts,'occurrence_status_counts':occurrence_counts,'categories_by_level':level_counts,
        'categories_by_control_status':control_counts,'categories_with_unambiguous_support':sum(bool(v) for v in safe_all.values()),
        'categories_with_unambiguous_primary_support':sum(bool(v) for v in safe_primary.values()),
        'categories_without_unambiguous_support':sorted(key for key,v in safe_all.items() if not v),
        'mechanical_findings':errors,'mechanical_validation_passed':not errors,'ready_to_freeze':False,'ready_for_mapper_training':False,
        'semantic_correctness_verified':False,'held_out_data_used':False,'new_model_calls':0,
        'support_caveat':'Candidate references from split mappings are not confirmed occurrence assignments; diagnostic B is not primary neutral training evidence.'}
    expanded=[]
    if not errors:
        for occurrence in occurrences:
            m=mapping[occurrence['inventory_id']]
            expanded.append({**occurrence,'mapping_status':m['status'],
                'canonical_id':m['canonical_ids'][0] if m['status']=='mapped' else None,
                'candidate_canonical_ids':m['canonical_ids'] if m['status']!='mapped' else [],
                'disambiguation_rule':m.get('disambiguation_rule'),
                'assignment_basis':'Unvalidated LLM vocabulary proposal; no change to source annotation or semantic claims'})
    return summary,expanded


def main(taxonomy_path,audit_dir,output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('Use a fresh audit output directory')
    invpath=audit_dir/'consolidation_upload/label_inventory.jsonl';occpath=audit_dir/'occurrences.private.jsonl'
    taxonomy=json.loads(taxonomy_path.read_text());inventory=read_lines(invpath);occurrences=read_lines(occpath)
    summary,expanded=check(taxonomy,inventory,occurrences)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if expanded:
        for name,rows in [('canonical_occurrences.private.jsonl',expanded),('needs_context.private.jsonl',[r for r in expanded if r['mapping_status']!='mapped'])]:
            (output_dir/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    report=taxonomy_path.with_name('consolidation_report_v2.md')
    sources=[taxonomy_path,invpath,occpath]+([report] if report.exists() else [])
    manifest={'input_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.iterdir() if p.is_file()}}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ('canonical_categories','mapping_status_counts','occurrence_status_counts','mechanical_validation_passed','ready_to_freeze')}))
    if summary['mechanical_findings']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxonomy',type=Path,default=ROOT/'research_runs/taxonomy_discovery_v1/responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json')
    parser.add_argument('--source-audit',type=Path,default=ROOT/'research_runs/taxonomy_discovery_v1/response_audit_v2')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();main(args.taxonomy,args.source_audit,args.output_dir)
