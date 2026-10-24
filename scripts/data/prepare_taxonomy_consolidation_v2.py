"""Audit optional B, normalize clause identities, and repackage actual label evidence."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

from audit_taxonomy_discovery import check_annotation,inventory,load,write_json,write_lines

ROOT=Path(__file__).resolve().parents[2]


def normalize_clause_ids(annotation):
    """Convert demonstrably sentence-local IDs to paragraph-wide IDs, with a log.

    Refuse ambiguous local duplicates or dependencies outside a sentence. Text,
    semantic IDs, labels, evidence and original response files are never edited.
    """
    result=copy.deepcopy(annotation);sentences=result['rhetorical_taxonomy']['sentence_structure']
    all_ids=[c['clause_id'] for s in sentences for c in s['clauses']]
    if len(all_ids)==len(set(all_ids)):return result,[]
    for sentence in sentences:
        local=[c['clause_id'] for c in sentence['clauses']]
        if len(local)!=len(set(local)):raise ValueError('Ambiguous duplicate clause ID within sentence')
        if any(dep not in local for c in sentence['clauses'] for dep in c['depends_on_clause_ids']):
            raise ValueError('Cannot resolve sentence-local clause dependency')
    changes=[];next_id=1
    for i,sentence in enumerate(sentences):
        mapping={clause['clause_id']:f'u{next_id+j}' for j,clause in enumerate(sentence['clauses'])}
        next_id+=len(mapping)
        for j,clause in enumerate(sentence['clauses']):
            new_id=mapping[clause['clause_id']];new_deps=[mapping[dep] for dep in clause['depends_on_clause_ids']]
            for field,new in (('clause_id',new_id),('depends_on_clause_ids',new_deps)):
                if clause[field]!=new:
                    changes.append({'annotation_id':annotation['paragraph_id'],'path':f'/rhetorical_taxonomy/sentence_structure/{i}/clauses/{j}/{field}',
                                    'before':copy.deepcopy(clause[field]),'after':copy.deepcopy(new),
                                    'reason':'Unambiguous sentence-local clause namespace converted to paragraph-wide IDs'})
                    clause[field]=new
    return result,changes


def main(output):
    import jsonschema
    base=ROOT/'research_runs/taxonomy_discovery_v1';responses=base/'responses';package=base/'package_v1';old_audit=base/'response_audit_v1'
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh consolidation audit directory')
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in responses.rglob('*') if p.is_file()}
    # A and neutral proposals were already audited; verify that source and checked bytes remain unchanged.
    old_manifest=json.loads((old_audit/'manifest.json').read_text())
    for path,digest in old_manifest['response_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Previously audited response changed')
    for path in ('original_annotations.checked.private.jsonl','neutral_proposals.checked.private.jsonl'):
        if hashlib.sha256((old_audit/path).read_bytes()).hexdigest()!=old_manifest['files'][path]:raise ValueError('Checked response changed')
    originals=load(old_audit/'original_annotations.checked.private.jsonl');neutral=load(old_audit/'neutral_proposals.checked.private.jsonl')
    optional=load(responses/'task_optional_B_annotations/annotations_B.jsonl')
    expected={r['paragraph_id']:r['text'] for r in load(package/'paragraphs_B.jsonl')}
    ids=[r['paragraph_id'] for r in optional]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Optional B ID coverage mismatch')
    validator=jsonschema.Draft202012Validator(json.loads((package/'annotation_output.schema.json').read_text()))
    normalized=[];changes=[];counts=collections.Counter();findings=[]
    for row in optional:
        validator.validate(row)
        if row['provenance']['annotation_context']!='single_paragraph':raise ValueError('Unexpected optional B context')
        fixed,log=normalize_clause_ids(row);changes.extend(log)
        validator.validate(fixed)
        errors,measure=check_annotation(fixed,expected[row['paragraph_id']]);counts.update(measure)
        findings.extend({'annotation_id':row['paragraph_id'],**error} for error in errors)
        normalized.append(fixed)
    if findings:raise ValueError('Optional B still has mechanical findings: '+json.dumps(findings))
    key=json.loads((package/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json').read_text())
    original_pairs={r['original_item_id']:r['pair_id'] for r in key};reference_pairs={r['reference_id']:r['pair_id'] for r in key}
    source_for_pair={r['pair_id']:r for r in load(package/'neutral_reference_pairs.jsonl')}
    proposals={r['pair_id']:r for r in neutral}
    annotations=[(r,'original_observed',original_pairs[r['paragraph_id']]) for r in originals]
    annotations.extend((r['proposed_neutral_annotation'],'neutral_proposed_paired',r['pair_id']) for r in neutral if r['proposed_neutral_annotation'])
    annotations.extend((r,'existing_reference_diagnostic',reference_pairs[r['paragraph_id']]) for r in normalized)
    catalog,occurrences=inventory(annotations)
    for row in catalog:
        positive={};diagnostic={}
        for context,n in row['context_counts'].items():
            (diagnostic if context=='existing_reference_diagnostic' else positive)[context]=n
        row['primary_context_counts']=positive;row['diagnostic_context_counts']=diagnostic
        row['diagnostic_only']=not positive
        row['support_policy']='Existing reference annotations are diagnostics, not independent extra training pairs or automatically accepted neutral examples'
    taxonomy=json.loads((responses/'task_3_consolidation/taxonomy_v0_1_proposal.json').read_text())
    incoming_review={'taxonomy_version':taxonomy.get('taxonomy_version'),'canonical_items':len(taxonomy.get('canonical_items',[])),
        'occurrence_mappings':len(taxonomy.get('occurrence_mapping',[])),
        'inventory_mappings':len(taxonomy.get('inventory_mapping',[])),
        'reported_result_type':taxonomy.get('consistency_checks',{}).get('result_type'),
        'accepted_as_discovery_consolidation':False,
        'reason':'Returned file explicitly reports missing completed annotations and no observed-label mappings; it is a design scaffold, not evidence-based consolidation'}
    # Retained proposals have identical text but may have different model annotations.
    comparisons=[]
    b_by_id={r['paragraph_id']:r for r in normalized}
    for pair_id,proposal in proposals.items():
        pair=source_for_pair[pair_id];b=b_by_id[pair['reference_id']];p=proposal['proposed_neutral_annotation']
        comparisons.append({'pair_id':pair_id,'decision':proposal['decision'],
            'reference_claims':len(b['semantic_content']['claims']),
            'proposal_claims':None if p is None else len(p['semantic_content']['claims']),
            'text_identical':proposal['proposed_neutral_text']==pair['plain_english_reference'],
            'same_text_annotations_identical':None if p is None else
                {k:v for k,v in b.items() if k not in ('paragraph_id','provenance')}=={k:v for k,v in p.items() if k not in ('paragraph_id','provenance')}})
    summary={'original_annotations':len(originals),'neutral_proposals':len(neutral),'existing_reference_diagnostic_annotations':len(normalized),
        'unique_underlying_training_pairs':len(key),'decision_counts':dict(collections.Counter(r['decision'] for r in neutral)),
        'optional_B_schema_valid':True,'optional_B_mechanical_findings':[], 'optional_B_counts':dict(counts),
        'normalized_B_records':len({change['annotation_id'] for change in changes}),'clause_id_fields_normalized':len(changes),
        'original_response_files_modified':False,'label_inventory_items':len(catalog),'label_occurrences':len(occurrences),
        'incoming_consolidation':incoming_review,'semantic_faithfulness_verified':False,'neutral_repairs_applied':False,
        'held_out_data_used':False,'new_model_calls':0,
        'retained_text_reannotation':{'identical_text_pairs':sum(r['text_identical'] for r in comparisons),
            'identical_annotations_excluding_id_and_provenance':sum(r['same_text_annotations_identical'] is True for r in comparisons if r['text_identical'])},
        'context_note':'84 annotations describe 28 underlying pairs. B diagnoses existing references; proposed-neutral annotations use paired context. Counts are not independent support.'}
    output.mkdir(parents=True,exist_ok=True);upload=output/'consolidation_upload';upload.mkdir()
    write_lines(output/'annotations_B.normalized.private.jsonl',normalized)
    write_lines(output/'normalization_log.private.jsonl',changes)
    write_lines(output/'occurrences.private.jsonl',occurrences)
    write_lines(output/'reference_proposal_comparison.private.jsonl',comparisons)
    write_json(output/'audit_summary.json',summary)
    write_json(upload/'discovery_summary.json',summary)
    write_lines(upload/'label_inventory.jsonl',catalog)
    prompt=(ROOT/'docs/taxonomy_discovery_v1/CONSOLIDATION_INVENTORY_PROMPT.md').read_text()
    supplement=('\n\n## This replacement consolidation request\n\n'
        'The inventory also includes existing_reference_diagnostic annotations (optional B). '
        'Use their diagnostic_context_counts and diagnostic_only flags. Do not count them as independent '
        'training pairs or automatically acceptable neutral examples. Prioritize original_observed and '
        'neutral_proposed_paired evidence; a diagnostic-only label may remain provisional. '
        'Different labels on identical retained text reveal annotation variability, not a style transformation.\n\n'
        'A previous response produced 38 design-only categories after reporting missing annotations. '
        'Do NOT reuse that scaffold as an empirical standard. The attached label_inventory.jsonl is the '
        'actual observed category evidence. Inventory IDs belong to THIS version and must not be joined '
        'to earlier inventory IDs. Begin by reporting how many inventory rows you actually read. '
        'If either attachment is unavailable, stop and list missing filenames; do not design a substitute taxonomy. '
        'Return taxonomy_v0_1_proposal_v2.json and consolidation_report_v2.md.\n')
    (upload/'CONSOLIDATION_PROMPT.md').write_text(prompt+supplement,encoding='utf-8')
    (output/'NEXT_STEP.md').write_text('''# Replacement consolidation upload

In a new ChatGPT conversation, paste `consolidation_upload/CONSOLIDATION_PROMPT.md`
and upload ONLY `label_inventory.jsonl` and `discovery_summary.json` from that folder.
These are completed annotation evidence, not the original paragraph input batches.

The earlier taxonomy is preserved but is not an empirical consolidation: it states
that annotations were missing and has no mappings. Ask for the new filenames in
the prompt; save them alongside the old file without replacing it.

Optional B is included as diagnostic evidence. Its clause IDs were unambiguously
renumbered in a separate derived copy, with every changed field logged. Text,
claims, labels and original responses were not edited.
''',encoding='utf-8')
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h for path,h in before.items()):raise ValueError('Source response changed')
    write_json(output/'manifest.json',{'input_response_hashes':before,'input_responses_unchanged':True,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ('existing_reference_diagnostic_annotations','normalized_B_records','label_inventory_items','label_occurrences')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
