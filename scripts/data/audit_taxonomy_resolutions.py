"""Audit Stage 1 proposals and copy an instruction-clarified Stage 2 packet."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def load(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def js(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def check_response(row,case,context):
    """Check identity, anchors and existing IDs; never certify semantic interpretation."""
    issues=[]
    for key in ('case_id','inventory_id','context_id'):
        if row.get(key)!=case[key]:issues.append({'code':'case_binding_mismatch','field':key})
    allowed=set(case['candidate_canonical_ids'])
    claims={r['claim_id'] for r in context['semantic_content']['claims']}
    relations={r['relation_id'] for r in context['semantic_content']['relations']}
    signatures=set()
    for i,assignment in enumerate(row['assignments']):
        if assignment['canonical_id'] not in allowed:issues.append({'code':'category_outside_case_candidates','assignment':i})
        if not set(assignment['claim_refs'])<=claims:issues.append({'code':'unknown_claim_ref','assignment':i})
        if not set(assignment['semantic_relation_refs'])<=relations:issues.append({'code':'unknown_relation_ref','assignment':i})
        for evidence in assignment['evidence']:
            if not evidence['quote'] or context['text'].count(evidence['quote'])<evidence['occurrence']:
                issues.append({'code':'unresolved_evidence_anchor','assignment':i})
        signature=json.dumps(assignment,sort_keys=True)
        if signature in signatures:issues.append({'code':'duplicate_component_assignment','assignment':i})
        signatures.add(signature)
    if not set(row['remaining_candidate_ids'])<=allowed:issues.append({'code':'remaining_candidate_outside_case'})
    return issues


def main(output):
    import jsonschema
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh audit directory')
    base=ROOT/'research_runs/taxonomy_discovery_v1';packet=base/'followup_v1';responses=base/'responses/task_4_context_resolution'
    source_manifest=json.loads((packet/'manifest.json').read_text())
    for name,digest in source_manifest['files'].items():
        if file_sha(packet/name)!=digest:raise ValueError('Prepared Stage 1/2 packet changed')
    response_files=sorted(path for path in responses.iterdir() if path.is_file())
    input_hashes={str(path.relative_to(ROOT)):file_sha(path) for path in response_files}
    annotations=[];findings=[];counts=collections.Counter();all_cases=[];batch_summaries=[];source_inputs=[]
    for batch in sorted((packet/'01_context_resolution').glob('batch_*')):
        number=batch.name.rsplit('_',1)[-1]
        result_path=responses/f'resolutions_{number}.jsonl';report_path=responses/f'resolution_report_{number}.md'
        if not result_path.exists():raise ValueError('Missing response file '+result_path.name)
        rows=load(result_path);cases={row['case_id']:row for row in load(batch/'cases.jsonl')}
        contexts={row['context_id']:row for row in load(batch/'contexts.jsonl')}
        schema=json.loads((batch/'resolution_output.schema.json').read_text());validator=jsonschema.Draft202012Validator(schema)
        source_inputs.extend(batch/path for path in ('cases.jsonl','contexts.jsonl','resolution_output.schema.json','candidate_categories.json'))
        ids=[row.get('case_id') for row in rows]
        if set(ids)!=set(cases) or len(ids)!=len(set(ids)):findings.append({'code':'batch_id_coverage_mismatch','batch':number})
        for row in rows:
            ident=row.get('case_id');errors=list(validator.iter_errors(row))
            for error in errors:findings.append({'case_id':ident,'code':'schema_error','path':list(error.absolute_path),'validator':error.validator})
            if errors or ident not in cases:continue
            case=cases[ident];context=contexts[case['context_id']]
            findings.extend({'case_id':ident,**error} for error in check_response(row,case,context))
            annotations.append(row);counts[row['status']]+=1
        all_cases.extend(cases)
        batch_summaries.append({'batch':number,'supplied':len(cases),'returned':len(rows),'report_present':report_path.exists()})
    if len(all_cases)!=len(set(all_cases)):findings.append({'code':'duplicate_case_across_batches'})
    output.mkdir(parents=True,exist_ok=True)
    group_summary={ident:{'statuses':dict(collections.Counter(row['status'] for row in annotations if row['inventory_id']==ident)),
        'proposed_categories':sorted({a['canonical_id'] for row in annotations if row['inventory_id']==ident for a in row['assignments']})}
        for ident in sorted({row['inventory_id'] for row in annotations})}
    summary={'expected_cases':len(all_cases),'returned_schema_valid_cases':len(annotations),'statuses':dict(counts),
        'batches':batch_summaries,'findings':findings,'mechanical_checks_passed':not findings,
        'sampled_cases_with_annotation_issues':sum(bool(row['annotation_issues']) for row in annotations),
        'sampled_cases_with_needs_review':sum(row['needs_review'] for row in annotations),
        'inventory_group_summary':group_summary,'semantic_judgments_independently_verified':False,
        'original_annotations_modified':False,'global_inventory_mappings_modified':False,
        'taxonomy_definitions_changed':False,'unsampled_occurrences_assigned':0,'new_model_calls':0,
        'stage2_decision':'hold_due_to_mechanical_findings' if findings else 'proceed_as_provisional_coverage_and_consistency_check_with_instruction_clarifications',
        'scope_note':'34 proposed classifications and five pending/gap decisions are sample-level; neither old semantic graphs nor full 720-case mapping is accepted as corrected.'}
    js(output/'audit_summary.json',summary)
    if not findings:
        (output/'checked_resolutions.private.jsonl').write_text(''.join(json.dumps({**row,'mechanical_checks_passed':True,'accepted_for_training':False},ensure_ascii=False)+'\n' for row in annotations))
        (output/'pending_cases.private.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in annotations if row['status'] in ('unresolved','taxonomy_gap')))
        # The catalogue and schemas remain identical; only procedural instructions change.
        destination=output/'stage2_ready';shutil.copytree(packet/'02_fresh_validation',destination)
        clarification=ROOT/'docs/taxonomy_validation_v1/STAGE2_CLARIFICATIONS.md'
        for batch in sorted(destination.glob('batch_*')):
            prompt=batch/'VALIDATION_PROMPT.md';prompt.write_text(prompt.read_text()+'\n\n'+clarification.read_text(),encoding='utf-8')
            for name in ('paragraphs.jsonl','taxonomy_categories.json','canonical_annotation_output.schema.json'):
                if file_sha(batch/name)!=file_sha(packet/'02_fresh_validation'/batch.name/name):raise ValueError('Stage 2 data, catalogue or schema changed')
        (output/'STAGE2_START_HERE.md').write_text('''# Stage 2 is ready: provisional vocabulary check

Use the updated `stage2_ready/batch_A`, `batch_B` and `batch_C` folders here.
The paragraphs, canonical catalogue and schema are unchanged from followup_v1;
VALIDATION_PROMPT.md adds generic procedural clarifications derived from Stage 1.

For each batch, start a NEW ChatGPT conversation. Paste VALIDATION_PROMPT.md and
upload paragraphs.jsonl, taxonomy_categories.json and
canonical_annotation_output.schema.json from that SAME batch folder. Do not show
prior responses, Stage 1 case material, this guide or the private pairing key.

Save canonical_annotations_A.jsonl and coverage_report_A.md (and corresponding
B/C names) under research_runs/taxonomy_discovery_v1/responses/task_5_fresh_validation/.
A/B together contain 14 distinct paragraphs from seven training pairs not used
for discovery. C has two masked repeats for a small consistency check; do not
identify them to the annotator or count them as new independent examples.

Stage 1 gave 30 single-category and four component proposals; three supplied
candidate sets did not fit, and two cases stayed unresolved. These do not prove
that new global categories are necessary. Keep gaps and ambiguities explicit
under the existing vocabulary. No prior semantic graph is treated as verified,
no taxonomy definition was changed, and no style operation is authorized.
''',encoding='utf-8')
    for name,digest in input_hashes.items():
        if file_sha(ROOT/name)!=digest:raise ValueError('Original response changed')
    extra=[ROOT/'docs/taxonomy_validation_v1/STAGE2_CLARIFICATIONS.md']+source_inputs
    js(output/'manifest.json',{'input_response_hashes':input_hashes,'responses_unchanged':True,
        'preparation_manifest_sha256':file_sha(packet/'manifest.json'),
        'additional_input_hashes':{str(path.relative_to(ROOT)):file_sha(path) for path in extra},
        'auditor_sha256':file_sha(Path(__file__).resolve()),
        'files':{str(path.relative_to(output)):file_sha(path) for path in output.rglob('*') if path.is_file()}})
    print(json.dumps({key:summary[key] for key in ('expected_cases','statuses','mechanical_checks_passed','stage2_decision')}))
    if findings:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
