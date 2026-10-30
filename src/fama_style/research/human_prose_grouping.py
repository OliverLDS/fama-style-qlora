"""Prepare five nonidentity development grouping triplets for manual inference.

Finite grouping only: original wording and target lexical controls stay private.
No training, generation, ML imports or reserved-content selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .author_comparison_corpus import file_sha
from .author_control_replication import verify_files
from .human_prose_review import occurrence_span, normalized
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'human_prose_grouping_v1'
CONDITIONS = ('ordinary', 'source_group_plan', 'oracle_target_group_plan')
PLAN_STATUSES = ('not_requested', 'followed', 'deviated_for_meaning',
                 'deviated_for_readability', 'other_deviation')
PROMPT = '''Rewrite each supplied source as one clear, compact paragraph of academic economics prose. Preserve substantive claims, actors, quantities, negation, uncertainty, conditions, causal direction and inferential force. Improve organization and expression where useful; do not invent facts, explanations or economic relationships. Use only the supplied source and source-quoted evidence. Keep unresolved external references unresolved. No particular author is requested. Do not force a wording-distance target or optimize for a detector.

All jobs include the same kind of source-bound claim evidence. Claim IDs are bookkeeping, not instructions to add relations. If no sentence_plan is supplied, choose the paragraph's sentence organization yourself.

If a sentence_plan is supplied, produce exactly one sentence for each listed group, in the listed order, covering that group's source-supported claim content. This is an explicit presentation request. Use natural clause combinations and punctuation, while preserving the original logical relations; grouping does not itself license a causal, explanatory or inferential link. A claim may contain several components spread across source sentences and therefore appear in more than one group: express the relevant components in their respective positions, without unnecessarily repeating the whole claim.

Keep the plan unless following it would change meaning or make the paragraph materially awkward. If it does, prioritize meaning and readability, make the smallest needed deviation, and report why. Do not resolve ambiguous referents by guessing or importing outside context. Minor wording changes are permitted; do not add a new mechanism or broaden an actor class merely to make the prose more fluent.

Return one valid JSON object per input, with exactly these four fields:
- job_id: copy the supplied ID.
- output: the rewritten paragraph only, without IDs, labels or commentary inside it.
- plan_status: not_requested if there is no plan; otherwise followed, deviated_for_meaning, deviated_for_readability, or other_deviation.
- plan_note: empty string when not_requested or followed; otherwise briefly explain the specific deviation and its reason.

Use plan_status as a report of what you did, not as a claim of semantic verification. Return no Markdown fences. Do not request, retrieve or compare against reference paragraphs, other batch answers, or earlier experiment results.'''


def groups(annotation, side):
    return [s['claim_ids'] for s in annotation[side]['sentences']]


def validate_annotation(a):
    """Check actionable groups and exact source evidence; repeated claims allowed."""
    ids=[c['claim_id'] for c in a['semantic_ledger']['claims']]
    namespace=set(ids)
    if not ids or len(ids)!=len(namespace): raise ValueError('claim_identity')
    for side,text in (('source_schema',a['plain_english']),('target_schema',a['original_paragraph'])):
        units=a[side]['sentences']
        if not units or normalized(' '.join(u['exact_text'] for u in units))!=normalized(text):
            raise ValueError('sentence_coverage')
        plan=groups(a,side)
        if any(not g or len(g)!=len(set(g)) or set(g)-namespace for g in plan):
            raise ValueError('empty_unknown_or_duplicate_group_claim')
        if set().union(*(set(g) for g in plan))!=namespace: raise ValueError('incomplete_group_coverage')
    for c in a['semantic_ledger']['claims']:
        if c['alignment_status']!='preserved': raise ValueError('annotator_uncertain_claim')
        if not c['rewrite_evidence'] or any(occurrence_span(a['plain_english'],e) is None for e in c['rewrite_evidence']):
            raise ValueError('source_evidence_not_exact')


def select_records(records, used):
    selected=[]
    for r in records:
        # Check split before examining any text or annotation fields.
        if r['split']!='development' or r['paragraph_id'] in used: continue
        if not r['schema_valid'] or r['audit']['errors'] or r['audit']['program_errors']:
            raise ValueError('invalid_unused_development_annotation')
        a=r['annotation']
        if groups(a,'source_schema')==groups(a,'target_schema'): continue
        validate_annotation(a)
        selected.append(r)
    selected.sort(key=lambda r:(r['paper_id'],r['paragraph_id']))
    if len(selected)!=5 or len({r['paragraph_id'] for r in selected})!=5 or len({r['paper_id'] for r in selected})!=3:
        raise ValueError('expected_five_unused_nonidentity_examples_three_papers')
    return selected


def payload(a, condition):
    if condition not in CONDITIONS: raise ValueError('unknown_condition')
    result={'source':a['plain_english'], 'claim_evidence':[
        {'claim_id':c['claim_id'],'source_quotes':[e['quote'] for e in c['rewrite_evidence']]}
        for c in a['semantic_ledger']['claims']]}
    if condition!='ordinary':
        result['sentence_plan']={'sentence_claim_groups':groups(a,
            'source_schema' if condition=='source_group_plan' else 'target_schema')}
    return result


def make_jobs(selected):
    batches=[[],[],[]];key=[]
    for i,r in enumerate(selected):
        a=r['annotation']
        for c,condition in enumerate(CONDITIONS):
            job_id=hashlib.sha256((PROTOCOL+':'+r['paragraph_id']+':'+condition).encode()).hexdigest()[:16]
            batch=(i+c)%3
            batches[batch].append({'job_id':job_id,**payload(a,condition)})
            key.append({'job_id':job_id,'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],
                'split':'development','condition':condition,'batch':batch+1,
                'source_sentence_count':len(groups(a,'source_schema')),
                'target_sentence_count':len(groups(a,'target_schema')),
                'source_needs_external_context':a['source_quality']['needs_external_context'],
                'annotation_needs_review':a['needs_review']})
    for batch in batches: batch.sort(key=lambda j:j['job_id'])
    return batches,key


def run(audit, previous_package, output):
    verify_files(audit);verify_files(previous_package)
    previous=json.loads((previous_package/'private_key.json').read_text())
    used={j['paragraph_id'] for j in previous}
    records=[json.loads(l) for l in (audit/'audits.private.jsonl').read_text().splitlines()]
    selected=select_records(records,used);batches,key=make_jobs(selected)
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','examples':5,'papers':3,
        'requested_outputs':15,'batches':3,'outputs_per_batch':5,'conditions':list(CONDITIONS),
        'all_source_target_groupings_differ':True,'previous_realization_examples_reused':0,
        'source_target_sentence_counts':sorted([{'source':len(groups(r['annotation'],'source_schema')),
            'target':len(groups(r['annotation'],'target_schema'))} for r in selected],key=lambda x:(x['source'],x['target'])),
        'examples_with_claims_spanning_source_sentences':sum(
            len([c for g in groups(r['annotation'],'source_schema') for c in g])>
            len({c for g in groups(r['annotation'],'source_schema') for c in g}) for r in selected),
        'semantic_tolerance_changed':False,'target_lexical_controls_supplied':False,
        'reference_wording_supplied':False,'oracle_not_deployment':True,'reserved_rows_used':0,
        'model_calls':0,'gpu_jobs_started':0,'training_called':False}
    fresh(output)
    for i,batch in enumerate(batches,1):
        directory=output/('batch_%02d'%i);directory.mkdir()
        write_lines(directory/'inputs.jsonl',batch)
        (directory/'TASK.md').write_text(PROMPT+'\n\nUse the attached inputs.jsonl. Save responses_%02d.jsonl.\n'%i)
    write_json(output/'private_key.json',key);write_json(output/'summary.json',summary)
    write_json(output/'output.schema.json',{'type':'object','required':['job_id','output','plan_status','plan_note'],
        'additionalProperties':False,'properties':{'job_id':{'type':'string','pattern':'^[a-f0-9]{16}$'},
        'output':{'type':'string','minLength':1},'plan_status':{'type':'string','enum':list(PLAN_STATUSES)},
        'plan_note':{'type':'string'}}})
    write_json(output/'selected_annotation_risks.private.json',[
        {'paragraph_id':r['paragraph_id'],'original_annotation_review_notes':r['annotation']['review_notes'],
         'source_quality':r['annotation']['source_quality'],
         'limited_assistant_reading_note':('The neutral text uses those people after mentioning a handful who know the cause; '
            'the original attaches the unidentified people clause to tens of thousands. Record this as a preexisting '
            'referent ambiguity; source-quoted claim evidence does not independently settle it. Do not silently repair the input.'
            if r['paragraph_id']=='HP_de362fe0aa7f21b7' else
            'External-context and annotation uncertainties remain; no exhaustive semantic verification performed.')}
        for r in selected])
    (output/'responses').mkdir()
    (output/'SESSION_NOTES.md').write_text('''# User session record — fill in if useful

Visible model label:
Thinking effort:
Chat arrangement/shared context:
Any regenerate, retry, edited output or failed batch:

Separate chats do not establish independence from automatic retrieval. Do not record credentials.
''')
    (output/'START_HERE.md').write_text('''# Five-example grouping diagnostic

Use GPT-6 with High thinking effort, the same visible setting as the previous check. There are three batches of five jobs, requesting 15 outputs total. Every batch has one variant of each paragraph, with a mixture of conditions. No batch contains two variants of the same paragraph.

For each batch:
1. Paste that batch's TASK.md and attach only its inputs.jsonl.
2. Save the first complete, unedited answer as responses_01.jsonl, responses_02.jsonl or responses_03.jsonl in this package's responses/ folder.
3. Keep the returned plan_status and plan_note fields. Do not correct the paragraph yourself; record any retry or failed batch.

Separate new chats are preferable for reducing direct carryover; they cannot guarantee independence from previous chats or retrieval. Keep model and effort settings consistent. SESSION_NOTES.md is optional; report any changes or shared context when returning the files. Do not attach private_key.json, original annotations, reference paragraphs, risk notes, earlier responses or results. No Kaggle upload or training is needed.

This compares ordinary rewriting with explicit source grouping and explicit oracle target grouping on five unused development examples from three papers. All targets differ from their source grouping. The only treatment difference is the grouping plan. All conditions have identical source text and source-quoted claim evidence. There are no target lexical/category controls. This tests grouping realization, not the entire three-level taxonomy or author-distinctive style. The oracle plan is reference-informed, not deployable.

Two source annotations spread a single claim across multiple sentences. The task explains how to preserve components without repeating the entire claim. Source quality remains uncertain; one input has a flagged referent ambiguity. Inputs are unchanged. Later review must separate input defects, generation drift, sentence/group adherence, copying, readability and expressive benefit; self-reported adherence is not a verified score.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'audit_manifest_sha256':file_sha(audit/'manifest.json'),
        'previous_package_manifest_sha256':file_sha(previous_package/'manifest.json'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()
            and p.name!='SESSION_NOTES.md'}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','previous-package','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.audit,a.previous_package,a.output),indent=2))


if __name__=='__main__':main()
