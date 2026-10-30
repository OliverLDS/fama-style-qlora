"""CPU preparation of a four-paper, source-grounded expression replication.

Source/paired annotations are fallible; output generation and training are absent.
"""
import argparse
import copy
import json
from pathlib import Path

from . import claim_bound_expression as prior
from . import claim_bound_expression_review as prior_review
from . import repaired_source_bindings as binding
from fama_style.rhetoric import operation_expansion as annotations
from fama_style.rhetoric import elementary_edits
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.prompting.economics import make_prompt, validate_evidence, validate_plan

ROOT = prior.ROOT
PROTOCOL = 'expression_replication_v1'
CONDITIONS = prior.CONDITIONS
OPERATIONS = {
 'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION':
    'Within the indicated output slot, use the supplied source focus as the grammatical starting point for the focus claim. '
    'Express the same supported relation or function. Keep any other claims in the slot and the grouping unchanged. '
    'Change presentation only, not attribution, perspective, scope or semantic strength.',
 'FINITE_SUPPORTED_CONSEQUENCE': prior.INSTRUCTIONS['FINITE_SUPPORTED_CONSEQUENCE']}
START = '''# Four additional papers: expression replication

Twelve first-attempt paragraphs in three four-job batches. Prepared only.
Use GPT-6 with High thinking effort consistently if available, matching the prior
reported setting. Record actual changes. Hidden temperature/seed are uncontrolled.

For XX = 01, 02, 03:
1. Prefer a separate new conversation with the same visible model/settings.
2. Attach ONLY batches/jobs_XX.jsonl and response.schema.json.
3. Paste TASK_PROMPT.md; keep the first returned responses_XX.jsonl unchanged.
4. Save it in research_runs/expression_replication_v1/responses/.

Each batch has one example per paper. No paired versions share a batch. Separate
chats cannot eliminate memory or establish independence; no chat deletion or
memory clearing is required. Record another arrangement if used. Optional
responses/notes.txt can record actual model, effort, dates and conversation setup.
Do not retry/select better outputs or silently fix partial/invalid response files.

Upload no private key, specification, eligibility ledger, evaluation plan,
references, PDF, reports, earlier answers or entire folder. No credentials belong
in prompts or responses. References remain local for evaluation only.

The primary comparison per case is grouping-only control versus expression
guidance with identical grouping and source-meaning reminders. New ordinary
outputs are secondary context. Two cases test theme, two test finite consequence.
One identity-grouping case is retained even if baseline performance is already
sufficient. These are inspected TRAINING examples, not a held-out benchmark or
source-only mapper. Tell Codex when the three response files are ready.
'''
EVALUATION = '''# Prespecified replication assessment

Freeze all twelve raw first attempts before inspecting outcomes. No best-of
selection or historical-control substitution. Primary contrast: selective
expression versus grouping-only control within each case; ordinary is secondary.
Use exact source/output evidence for judgments, not reference wording in prompts.

Evaluate operation realization, grouping, local claim fidelity, nonpacking
style/clarity and copying separately. Distinguish realization from actual quality
improvement and from author specificity. A baseline may already realize the
requested form; retain and report such ceiling cases, not new replacements.
Meaning reminders are identical across arms. Any resulting fidelity improvement
cannot be uniquely credited to the expression operation. Modest drift can be
weighed against worthwhile gains with explicit changed claims and benefits.

F1965 theme: the source distribution becomes the grammatical starting point,
while statistical independence and the information implication retain scope.
Grouping is identity KEEP/KEEP; don't misreport it as a packing improvement.
ME1997 theme: the modeled economic variable becomes the grammatical starting
point without dropping BSV attribution, two biases, actual random-walk earnings
or investors' mistaken regime beliefs. Regime likelihood and corrective sequences
must retain their conditional/perceived status, not become actual regimes.
FF2000 finite consequence: preserve rate/cohort/year bindings and near equality;
realize the unchanged payer-share consequence finitely within its merged slot.
Keep counts and shares separate, including the opposite delisting directions.
VL2018 finite consequence: the negative-premium probability consequence retains
its source actor/meaning in the existing volatility/consequence/tail slot. Its
scope is c2/c3 only; c4 must survive unchanged in meaning. Preserve chance,
three/five-year investment-evaluation context, ten/20-year nontrivial risk and
qualitative probabilities. No invented numerical risk, guaranteed loss or
elimination of long-horizon risk.

Report per-case and per-operation outcomes, including negative/mixed results.
Lexical overlap/token edits are diagnostics, not semantic or author-style scores.
Do not interpret a source-consistent paraphrase as failure just because it differs
from the reference. Off-target wording changes, hidden decoding, chat/batch memory,
prior source familiarity and unblinded assistant assessment limit causality.

Two additional papers per operation support a bounded mechanism replication,
not population estimates, a distinctive-author classifier or deployment evidence.
Do not fit a mapper or adapter from these outputs automatically. No reserved test
or validation examples are used. No GPU job is authorized by this package.
'''


def compile_operation(card, spec):
    claims = {c['claim_id']:c for c in card['evidence']['claim_index']}
    refs = spec['claim_refs']
    if not refs or len(set(refs)) != len(refs) or not set(refs) <= set(claims): raise ValueError('operation_claim_refs')
    slots = [s['claim_refs'] for p in card['plan'] for s in p['output_slots']]
    matches = [i for i,s in enumerate(slots) if set(refs) <= set(s)]
    if len(matches) != 1: raise ValueError('operation_scope_must_share_one_slot')
    slot = slots[matches[0]]
    if [r for r in slot if r in refs] != refs: raise ValueError('operation_scope_order')
    source = card['record']['source']; control = spec['control_id']
    operation = {'control_id':control,'claim_refs':refs,'output_slot':matches[0]+1,
                 'slot_claim_refs':slot,'untargeted_slot_claim_refs':[c for c in slot if c not in refs]}
    def owned(cid, quote):
        if cid not in refs or len(claims[cid]['source_spans']) != 1: raise ValueError('single_owned_anchor_required')
        return binding.anchored(source,quote,claims[cid]['source_spans'][0])[1]
    if control == 'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION':
        operation.update(focus_claim=spec['focus_claim'],focus_source_span=owned(spec['focus_claim'],spec['focus_quote']))
    elif control == 'FINITE_SUPPORTED_CONSEQUENCE':
        if len(refs) != 2 or spec['actor_claim'] != refs[-1]: raise ValueError('one_antecedent_one_consequence_required')
        edges = [r for r in card['evidence']['source_relations'] if r['relation_id'] == spec['relation_id']]
        if len(edges) != 1: raise ValueError('missing_source_edge')
        edge = edges[0]
        if edge['kind'] != 'causal' or edge['certainty'] != 'explicit' or edge['from_claim_refs'] != refs[:1] or edge['to_claim_refs'] != refs[1:]:
            raise ValueError('explicit_source_causal_edge_required')
        endpoints = [r for r in card['endpoint_bindings'] if r['relation_id'] == edge['relation_id']]
        if len(endpoints) != 1: raise ValueError('endpoint_bindings_required')
        for side,cid in (('from',refs[0]),('to',refs[1])):
            e = endpoints[0][side]
            if e['claim_id'] != cid or owned(cid,e['span']['quote']) != e['span']: raise ValueError('misbound_endpoint')
        operation.update(relation_id=edge['relation_id'],actor_claim=spec['actor_claim'],
                         actor_source_span=owned(spec['actor_claim'],spec['actor_quote']))
    else: raise ValueError('unsupported_operation')
    operation['instruction'] = OPERATIONS[control]
    return operation


def build_card(info, annotation, spec):
    if info['split'] != 'train' or info['case_id'] != spec['parent_case_id'] or info['paper_key'] != spec['paper_key']: raise ValueError('wrong_parent_or_split')
    if info['paper_key'] in {'FJ1983','F1980'}: raise ValueError('additional_papers_required')
    record = info['record']; checked = annotations.validate_annotation(record,annotation)
    if annotation['status'] != 'already_structured': raise ValueError('accepted_grounded_annotation_required')
    claim_map = {c['claim_id']:c for c in checked['compiled_claims']}
    # Re-resolve endpoint-owned evidence, not legacy character offsets. The old
    # annotation is never edited. Interpretation/explicit certainty is retained.
    relations=[]; endpoints=[]
    for r in annotation['relations']:
        if len(r['from_claim_refs']) != 1 or len(r['to_claim_refs']) != 1: raise ValueError('review_multi_endpoint_relation_first')
        evidence=[]; anchors={}
        for side in ('from','to'):
            cid=r[side+'_claim_refs'][0]; spans=claim_map[cid]['source_spans']
            if len(spans) != 1: raise ValueError('review_multispan_claim_first')
            ev,span=binding.anchored(record['source'],spans[0]['quote'],spans[0]);evidence.append(ev)
            anchors[side]={'claim_id':cid,'span':span}
        relations.append({k:r[k] for k in ('relation_id','kind','from_claim_refs','to_claim_refs','certainty')}|{'source_evidence':evidence})
        endpoints.append({'relation_id':r['relation_id'],**anchors})
    reviewed=copy.deepcopy(annotation);reviewed['relations']=relations
    try:
        plan=elementary_edits.compile_edits(record,reviewed)['generation_payload']['edits']
        grouping_origin='reference_informed_oracle'
    except ValueError as error:
        if str(error) != 'no_elementary_packing_change': raise
        source=elementary_edits.partition(reviewed['source_units'],set(claim_map))
        target=elementary_edits.partition(reviewed['target_units'],set(claim_map))
        if source != target: raise ValueError('identity_grouping_not_identical')
        plan=[{'op':'KEEP','source_unit_ids':[u['unit_id']],'output_slots':[{'claim_refs':u['claim_refs']}]} for u in reviewed['source_units']]
        grouping_origin='source_identity_checked_against_reference'
    index=[{k:c[k] for k in ('claim_id','composite','epistemic_status','source_spans')}|
           {'protected':[{'kind':p['kind'],'source_spans':annotations.resolve(record['source'],p['source_evidence'])} for p in c['protected']]}
           for c in checked['compiled_claims']]
    evidence={'source_units':record['source_units'],'claim_index':index,'source_relations':relations}
    validate_evidence(record['source'],evidence);validate_plan(record['source'],evidence,plan)
    card={'id':info['case_id'],'source_id':info['source_id'],'paper_key':info['paper_key'],'split':'train',
          'record':copy.deepcopy(record),'annotation':reviewed,'evidence':evidence,'plan':plan,'endpoint_bindings':endpoints,
          'source_sha256':annotations.sha(record['source']),'target_sha256':annotations.sha(record['reference']),
          'grouping_origin':grouping_origin,'semantic_guard':spec['semantic_guard'],
          'eligibility_note':spec['eligibility_note'],'source_reference_caveat':spec['source_reference_caveat'],
          'source_review_method':'In-session source/paired evidence inspection; not independent verification.',
          'training_ready':False}
    card['expression_operation']=compile_operation(card,spec)
    return card


def load_inputs(spec_path):
    base=ROOT/'research_runs/operation_expansion_v1';package=base/'package_v1';audit=base/'response_audit_v1'
    key=annotations.check_package(package);manifest=json.loads((audit/'manifest.json').read_text())
    binding.verify_files(audit,manifest['files'])
    infos={r['case_id']:r for r in key['records']};checked={r['case_id']:r['annotation'] for r in binding.read_lines(audit/'annotations.checked.private.jsonl')}
    previous=ROOT/'research_runs/claim_bound_expression_v1';prior.check_package(previous/'package_v1')
    verification=json.loads((previous/'review_verification.json').read_text());review_path=previous/'review_v1/manifest.json'
    if binding.file_sha(review_path) != verification['review_manifest_sha256']: raise ValueError('prior_review_changed')
    review_manifest=json.loads(review_path.read_text());binding.verify_files(previous/'review_v1',review_manifest['files'])
    if review_manifest['code_sha256'] != binding.file_sha(Path(prior_review.__file__)): raise ValueError('prior_review_code_changed')
    spec=json.loads(spec_path.read_text())
    if spec['protocol'] != PROTOCOL or len(spec['records']) != 4 or len({r['parent_case_id'] for r in spec['records']}) != 4: raise ValueError('four_distinct_cases_required')
    if {r['paper_key'] for r in spec['records']} != {'F1965','ME1997','FF2000','VL2018'}: raise ValueError('four_additional_papers_required')
    controls=[r['control_id'] for r in spec['records']]
    if any(controls.count(c) != 2 for c in OPERATIONS) or not set(controls)<=set(OPERATIONS): raise ValueError('two_cases_per_operation_required')
    cards=[build_card(infos[s['parent_case_id']],checked[s['parent_case_id']],s) for s in sorted(spec['records'],key=lambda s:s['paper_key'])]
    return cards,{'annotation_package_manifest_sha256':binding.file_sha(package/'manifest.json'),
        'annotation_audit_manifest_sha256':binding.file_sha(audit/'manifest.json'),'frozen_train_sha256':binding.TRAIN_SHA,
        'specification_sha256':binding.file_sha(spec_path),'prior_expression_review_manifest_sha256':binding.file_sha(review_path)}


def prompt_for(card,condition):
    if condition not in CONDITIONS: raise ValueError('unknown_condition')
    prompt=make_prompt(card['record']['source'],'ordinary' if condition=='ordinary' else 'planned',card['evidence'],None if condition=='ordinary' else card['plan'])
    tail='\nReturn only the rewritten paragraph.'
    if not prompt.endswith(tail): raise ValueError('prompt_template_changed')
    body=prompt[:-len(tail)]+'\nShared source-meaning reminders (all conditions):\n'+card['semantic_guard']
    if condition=='selective_expression':body+='\nClaim-bound expression operation (keep the grouping plan unchanged):\n'+json.dumps(card['expression_operation'],ensure_ascii=False,sort_keys=True)
    return body+tail


def build_jobs(cards):
    if len(cards)!=4 or len({c['paper_key'] for c in cards})!=4: raise ValueError('four_paper_design_required')
    cards=sorted(cards,key=lambda c:annotations.sha(PROTOCOL+'|'+c['id']));batches=[];mapping=[]
    for b in range(3):
        jobs=[]
        for i,c in enumerate(cards):
            condition=CONDITIONS[(b+i)%3];prompt=prompt_for(c,condition)
            if c['record']['reference'] in prompt: raise ValueError('full_reference_leakage')
            job={'job_id':'J'+annotations.sha(PROTOCOL+'|'+c['id']+'|'+condition)[:16],'prompt_sha256':annotations.sha(prompt),'prompt':prompt}
            jobs.append(job);mapping.append({k:job[k] for k in ('job_id','prompt_sha256')}|
                {'source_id':c['id'],'paper_key':c['paper_key'],'condition':condition,
                 'control_id':c['expression_operation']['control_id'],'batch_file':f'jobs_{b+1:02d}.jsonl','response_file':f'responses_{b+1:02d}.jsonl'})
        batches.append(sorted(jobs,key=lambda j:annotations.sha(j['job_id'])))
    return batches,mapping


def prepare(spec_path,output):
    cards,provenance=load_inputs(spec_path);batches,mapping=build_jobs(cards)
    fresh(output);(output/'batches').mkdir()
    for i,batch in enumerate(batches,1):write_lines(output/'batches'/f'jobs_{i:02d}.jsonl',batch)
    write_json(output/'private_key.private.json',{'protocol':PROTOCOL,**provenance,'rows':cards,'jobs':mapping,
        'selection':'Four pre-output source-reviewed training cases; two per operation, excluding both initial papers.',
        'operation_scope_version':'v1.1_subset_of_slot_and_identity_grouping','planned_model':'GPT-6','planned_thinking_effort':'High',
        'actual_generation_metadata':'pending','hidden_decoding_controlled':False})
    write_json(output/'eligibility.private.json',[{k:c[k] for k in ('id','source_id','paper_key','split','eligibility_note','source_reference_caveat','grouping_origin','source_sha256','expression_operation')} for c in cards])
    for name,text in (('START_HERE.md',START),('TASK_PROMPT.md',prior.parent.TASK),('EVALUATION_PLAN.md',EVALUATION)):(output/name).write_text(text,encoding='utf-8')
    write_json(output/'response.schema.json',prior.parent.parent.RESPONSE_SCHEMA)
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','papers':4,'requested_outputs':12,'batches':3,'jobs_per_batch':4,
        'conditions':list(CONDITIONS),'operation_cases':{c:2 for c in OPERATIONS},'initial_papers_excluded':['F1980','FJ1983'],
        'cases':[{'paper_key':c['paper_key'],'control_id':c['expression_operation']['control_id'],'grouping_origin':c['grouping_origin'],
                  'operation_targets_subset_of_slot':bool(c['expression_operation']['untargeted_slot_claim_refs'])} for c in cards],
        'semantic_accuracy':None,'author_style_accuracy':None,'new_model_calls':0,'gpu_jobs_started':0,'training_called':False,
        'validation_rows_read':0,'reserved_test_rows_read':0}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,**provenance,'code_sha256':binding.file_sha(Path(__file__)),
        'specification_path':str(spec_path.resolve().relative_to(ROOT)),
        'prompt_builder_sha256':binding.file_sha(ROOT/'src/fama_style/prompting/economics.py'),
        'files':{str(p.relative_to(output)):binding.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print('Prepared 12 jobs across four additional training papers. No model calls or training.')


def check_package(package):
    manifest=json.loads((package/'manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['code_sha256']!=binding.file_sha(Path(__file__)): raise ValueError('replication_compiler_changed')
    if manifest['prompt_builder_sha256']!=binding.file_sha(ROOT/'src/fama_style/prompting/economics.py'): raise ValueError('prompt_builder_changed')
    binding.verify_files(package,manifest['files']);cards,provenance=load_inputs(ROOT/manifest['specification_path']);batches,mapping=build_jobs(cards)
    key=json.loads((package/'private_key.private.json').read_text())
    if key['rows']!=cards or key['jobs']!=mapping or any(key[k]!=v or manifest[k]!=v for k,v in provenance.items()): raise ValueError('replication_parent_changed')
    if sorted(p.name for p in (package/'batches').iterdir())!=[f'jobs_{i:02d}.jsonl' for i in range(1,4)]:raise ValueError('unexpected_batch_files')
    for i,batch in enumerate(batches,1):
        if binding.read_lines(package/'batches'/f'jobs_{i:02d}.jsonl')!=batch:raise ValueError('replication_jobs_changed')
    return key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specification',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--check-package',type=Path)
    a=p.parse_args()
    if a.check_package:
        if a.specification or a.output_dir:p.error('Check or prepare, not both')
        check_package(a.check_package);print('Replication package and parent hashes verified.')
    elif a.specification and a.output_dir:prepare(a.specification,a.output_dir)
    else:p.error('Provide specification and output-dir, or check-package')


if __name__=='__main__':main()
