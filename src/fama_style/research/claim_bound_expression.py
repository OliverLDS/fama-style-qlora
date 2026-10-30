"""Prepare six source-grounded expression probes; no training or model calls."""
import argparse
import copy
import json
from pathlib import Path

from . import source_repair_comparison as parent
from . import source_repair_review as review
from . import repaired_source_bindings as binding
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

ROOT = parent.ROOT
PROTOCOL = 'claim_bound_expression_v1'
CONDITIONS = ('ordinary', 'grouping_control', 'selective_expression')
INSTRUCTIONS = {
 'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION':
    'Within the indicated output slot, use the supplied source focus as the grammatical starting point for the focus claim. '
    'Express the same already-supported structural relationship and attach the other claim in the slot. '
    'Keep the grouping unchanged. This changes presentation only, not the relationship or its strength.',
 'FINITE_SUPPORTED_CONSEQUENCE':
    'Within the indicated output slot, give the consequence claim its own explicit source actor and finite predicate, '
    'rather than expressing it only through a participial result. Keep it in the SAME sentence as its antecedent claims. '
    'Keep the grouping unchanged. This changes presentation only, not the causal direction or certainty.'}
SEMANTIC_GUARDS = {
 'FJ1983': 'Keep the stated common-core relationship and all-types scope; add no importance ranking or decision stage. '
           'Preserve actor roles, the ratification definition and stage sequence, large-organization scopes, and mutual monitoring. '
           'Keep the explicit origin of monitoring in collaborative production/development, rather than mere co-occurrence.',
 'F1980': 'Keep the rental asset, supplier/recipient, rate, market signal and qualified dependence. '
          'Preserve the management function and information-using role of the labor market. '
          'Keep uncertainty about current pay separate from the asserted future-wage effect. '
          'Retain the explicit future-wage-to-manager-stake relation with the same actors and direction; add no cause, timing or condition.'}
START = '''# Claim-bound expression check: six outputs, no training

Use GPT-6 with High thinking effort consistently, matching the prior reported
setting if available. Record any actual change. Hidden decoding is uncontrolled.

For each XX = 01, 02, 03:
- Prefer a separate new conversation using the same visible model/settings.
- Attach ONLY batches/jobs_XX.jsonl and response.schema.json.
- Paste TASK_PROMPT.md and keep the first returned responses_XX.jsonl unchanged.
- Save it in research_runs/claim_bound_expression_v1/responses/.

Each batch has two different papers. Paired versions never share a batch.
Separate chats do not eliminate memory. No deletion or memory clearing is needed.
If using another arrangement, record it; no independence claim is made. Optional
responses/notes.txt can record model, visible effort, dates and chat arrangement.
Keep partial/invalid first attempts; do not select better retries or repair files.

Do not upload this file, the private key, specification, reports, reference text,
PDFs, old answers or entire folders. No credentials belong in any prompt or reply.
The model receives source text/evidence, an optional oracle grouping plan, and at
most one claim-bound expression operation. Original references stay local.

Two new ordinary rewrites are context baselines. The PRIMARY contrast is the
new grouping-only control versus expression guidance with IDENTICAL grouping.
Identical source-meaning reminders appear in all three conditions, including the
previously weakened monitoring-origin relation. These are fresh controls, not
reused historical ordinary prompts. The two cases test different operations; neither is an author-style benchmark
or independent replication. After saving all three response files, tell Codex.
'''
EVALUATION = '''# Prespecified assessment

Freeze all raw replies first. Evaluate all six first attempts, including failures.
The primary paired contrast per paper is selective_expression minus
grouping_control. Semantic reminders are shared across ALL conditions, so a
reminder-induced fidelity gain is not attributed to the expression operation. Ordinary rewriting is a secondary organization baseline; prior
outputs are historical context, never concurrent controls.

FJ1983: inspect whether the source-bound decision-control focus is actually the
grammatical starting point, and whether the core relation, universality, hierarchy
participants, ratification definition/sequence, large scopes, mutual monitoring
and its origin in production/human-capital development remain intact.

F1980: inspect whether c6 has an explicit actor and finite predication within the
existing c5/c6 sentence, and whether it remains linked to the asserted future-wage
effect rather than the qualified current-pay clause. Check rental roles, rate,
market signal, likelihood, management function and information use separately.
A correct count or the appearance of a connective alone is not realization.

Assess operation realization, source fidelity, nonpacking organization/style and
copying separately. Allow modest drift to be weighed against worthwhile gains,
with exact changed claims and benefits disclosed. Reference comparison is
evaluation-only; do not reward a wording match that changes source meaning.
Use exact evidence for judgments and separate in-session assessments from
mechanical checks. No held-out, author-specific or independent-verifier claims.

These two source/reference-informed oracle plans do not establish deployment
performance. If expression guidance has no useful gain, revisit realization and
controls before fitting a mapper or adapter. If it helps, replicate across more
eligible papers and obtain controlled comparison prose before claiming author
specificity. No automatic extension, generation or GPU training is authorized.
'''


def compile_operation(card, spec):
    if card['paper_key'] != spec['paper_key'] or card['input_variant'] != 'repaired': raise ValueError('wrong_operation_case')
    if card['context_flags'] or card['alignment_caveats']: raise ValueError('bounded_case_required')
    claims={c['claim_id']:c for c in card['evidence']['claim_index']}
    refs=spec['claim_refs']
    if not refs or len(set(refs))!=len(refs) or not set(refs)<=set(claims): raise ValueError('operation_claim_refs')
    slots=[s['claim_refs'] for op in card['plan'] for s in op['output_slots']]
    matches=[i for i,s in enumerate(slots) if s==refs]
    if len(matches)!=1: raise ValueError('operation_requires_fixed_shared_slot')
    control=spec['control_id']; source=card['record']['source']
    operation={'control_id':control,'claim_refs':refs,'output_slot':matches[0]+1}
    if control=='FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION':
        cid=spec['focus_claim']
        if cid not in refs: raise ValueError('focus_outside_operation')
        owners=claims[cid]['source_spans']
        if len(owners)!=1: raise ValueError('single_focus_owner_required')
        _,anchor=binding.anchored(source,spec['focus_quote'],owners[0])
        operation.update(focus_claim=cid,focus_source_span=anchor)
    elif control=='FINITE_SUPPORTED_CONSEQUENCE':
        edges=[r for r in card['evidence']['source_relations'] if r['relation_id']==spec['relation_id']]
        if len(edges)!=1: raise ValueError('missing_source_edge')
        edge=edges[0]
        if edge['certainty']!='explicit' or edge['kind']!='causal' or edge['from_claim_refs']!=refs[:-1] or edge['to_claim_refs']!=refs[-1:]:
            raise ValueError('unsupported_consequence_edge')
        cid=spec['actor_claim']
        if cid!=refs[-1] or len(claims[cid]['source_spans'])!=1: raise ValueError('actor_must_belong_to_consequence')
        _,anchor=binding.anchored(source,spec['actor_quote'],claims[cid]['source_spans'][0])
        endpoints=[r for r in card['endpoint_bindings'] if r['relation_id']==spec['relation_id']]
        if len(endpoints)!=1: raise ValueError('missing_endpoint_binding')
        for side,known in (('from',refs[0]),('to',cid)):
            part=endpoints[0][side]
            if part['claim_id']!=known or not any(o['start']<=part['span']['start']<part['span']['end']<=o['end'] for o in claims[known]['source_spans']):
                raise ValueError('misbound_consequence_endpoint')
            binding.anchored(source,part['span']['quote'],part['span'])
        operation.update(relation_id=edge['relation_id'],actor_claim=cid,actor_source_span=anchor)
    else: raise ValueError('unsupported_operation')
    operation['instruction']=INSTRUCTIONS[control]
    return operation


def load_inputs(spec_path):
    root=ROOT/'research_runs/source_repair_comparison_v1'
    key=parent.check_package(root/'package_v1')
    verification=json.loads((root/'review_verification.json').read_text())
    manifest=root/'review_v1/manifest.json'
    if binding.file_sha(manifest)!=verification['review_manifest_sha256']: raise ValueError('review_manifest_changed')
    data=json.loads(manifest.read_text()); binding.verify_files(root/'review_v1',data['files'])
    if data['code_sha256']!=binding.file_sha(Path(review.__file__)): raise ValueError('review_implementation_changed')
    spec=json.loads(spec_path.read_text())
    if spec['protocol']!=PROTOCOL or len(spec['records'])!=2 or {s['paper_key'] for s in spec['records']}!={'FJ1983','F1980'}: raise ValueError('two_fixed_cases_required')
    cards=[]
    for s in sorted(spec['records'],key=lambda s:s['paper_key']):
        card=copy.deepcopy(next(c for c in key['rows'] if c['paper_key']==s['paper_key'] and c['input_variant']=='repaired'))
        card['expression_operation']=compile_operation(card,s);cards.append(card)
    return cards,{'source_comparison_package_sha256':binding.file_sha(root/'package_v1/manifest.json'),
                  'parent_review_manifest_sha256':binding.file_sha(manifest),'specification_sha256':binding.file_sha(spec_path),
                  'control_catalogue_sha256':binding.file_sha(ROOT/'docs/residual_style_audit_v1_controls.json')}


def prompt_for(card,condition):
    if condition not in CONDITIONS: raise ValueError('unknown_condition')
    mode='ordinary' if condition=='ordinary' else 'planned'
    prompt=parent.prompt_for(card,mode)
    tail='\nReturn only the rewritten paragraph.'
    if not prompt.endswith(tail): raise ValueError('prompt_tail_changed')
    prompt=prompt[:-len(tail)]+'\nShared source-meaning reminders (all conditions):\n'+SEMANTIC_GUARDS[card['paper_key']]+tail
    if condition=='selective_expression':
        prompt=prompt[:-len(tail)]+'\nClaim-bound expression operation (keep the grouping plan unchanged):\n'+json.dumps(card['expression_operation'],ensure_ascii=False,sort_keys=True)+tail
    return prompt


def build_jobs(cards):
    if len(cards)!=2 or {c['paper_key'] for c in cards}!={'FJ1983','F1980'}: raise ValueError('two_fixed_cases_required')
    cards=sorted(cards,key=lambda c:c['paper_key']);batches=[];mapping=[]
    for b in range(3):
        batch=[]
        for i,card in enumerate(cards):
            condition=CONDITIONS[(b+i)%3];prompt=prompt_for(card,condition)
            if card['record']['reference'] in prompt: raise ValueError('full_reference_leakage')
            job={'job_id':'J'+parent.base.sha(PROTOCOL+'|'+card['id']+'|'+condition)[:16],
                 'prompt_sha256':parent.base.sha(prompt),'prompt':prompt}
            batch.append(job);mapping.append({k:job[k] for k in ('job_id','prompt_sha256')}|
                {'source_id':card['id'],'paper_key':card['paper_key'],'condition':condition,
                 'batch_file':f'jobs_{b+1:02d}.jsonl','response_file':f'responses_{b+1:02d}.jsonl'})
        batches.append(sorted(batch,key=lambda j:parent.base.sha(j['job_id'])))
    return batches,mapping


def prepare(spec_path,output):
    cards,provenance=load_inputs(spec_path);batches,mapping=build_jobs(cards)
    fresh(output);(output/'batches').mkdir()
    for i,batch in enumerate(batches,1): write_lines(output/'batches'/f'jobs_{i:02d}.jsonl',batch)
    write_json(output/'private_key.private.json',{'protocol':PROTOCOL,**provenance,'rows':cards,'jobs':mapping,
        'planned_model':'GPT-6','planned_thinking_effort':'High','actual_generation_metadata':'pending',
        'primary_contrast':'selective_expression versus grouping_control within each paper',
        'reference_informed_grouping':True,'hidden_decoding_controlled':False})
    for name,text in (('START_HERE.md',START),('TASK_PROMPT.md',parent.TASK),('EVALUATION_PLAN.md',EVALUATION)):
        (output/name).write_text(text,encoding='utf-8')
    write_json(output/'response.schema.json',parent.parent.RESPONSE_SCHEMA)
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','requested_outputs':6,'papers':2,'batches':3,
        'conditions':list(CONDITIONS),'grouping_fixed_in_primary_contrast':True,
        'operations':[{'paper_key':c['paper_key'],'control_id':c['expression_operation']['control_id']} for c in cards],
        'label_reuse_tested':False,'model_calls':0,'gpu_jobs_started':0,'training_called':False,
        'validation_rows_read':0,'reserved_test_rows_read':0,'semantic_accuracy':None,'author_style_accuracy':None}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,**provenance,'code_sha256':binding.file_sha(Path(__file__)),
        'specification_path':str(spec_path.resolve().relative_to(ROOT)),
        'files':{str(p.relative_to(output)):binding.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print('Prepared three two-job batches: six outputs; no model call or training.')


def check_package(package):
    manifest=json.loads((package/'manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['code_sha256']!=binding.file_sha(Path(__file__)): raise ValueError('expression_compiler_changed')
    binding.verify_files(package,manifest['files'])
    cards,provenance=load_inputs(ROOT/manifest['specification_path']);batches,mapping=build_jobs(cards)
    key=json.loads((package/'private_key.private.json').read_text())
    if key['rows']!=cards or key['jobs']!=mapping or any(key[k]!=v or manifest[k]!=v for k,v in provenance.items()): raise ValueError('expression_inputs_changed')
    if sorted(p.name for p in (package/'batches').iterdir())!=[f'jobs_{i:02d}.jsonl' for i in range(1,4)]: raise ValueError('unexpected_batch_files')
    for i,batch in enumerate(batches,1):
        if parent.read_lines(package/'batches'/f'jobs_{i:02d}.jsonl')!=batch: raise ValueError('expression_jobs_changed')
    return key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--check-package',type=Path)
    args=parser.parse_args()
    if args.check_package:
        if args.specification or args.output_dir:parser.error('Check or prepare, not both')
        check_package(args.check_package);print('Expression package and parent hashes verified.')
    elif args.specification and args.output_dir:prepare(args.specification,args.output_dir)
    else:parser.error('Provide specification and output-dir, or check-package')


if __name__=='__main__':main()
