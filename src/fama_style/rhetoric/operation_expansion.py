"""Training-only operation-family expansion: candidate export and grounded-plan audit."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
from .concise_plan import TRAIN_SHA,file_sha,read_lines
from .experiment import fresh,write_json,write_lines
from . import bound_grouping as grouping
from . import grounded_operations as grounded
from . import strong_realizer_check as transport
from fama_style.training.plan_realizer import COMMON
from fama_style.research.claims import sha,risk_flags
from fama_style.research.signal import overlap

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='operation_expansion_v1'
FAMILY_VERSION='operation_families_v2_candidate_extensions'
FAMILIES={
 'premise_inference_scope':{'seed_operation':'separate_premise_inference_qualification',
  'description':'Separate an existing premise, an already-supported inference, and its applicability condition or qualification. Do not turn an association into a cause.',
  'roles':['premise','inference','scope','qualification','supporting_detail'],
  'required_roles':['premise','inference'],
  'language_controls':['retain_modal_strength','mark_existing_inference','mark_existing_scope_contrast'],
  'cue_patterns':[r'\bif\b|\bunless\b|\bwhen\b|\bprovided\b',r'\bthus\b|\btherefore\b|\bhence\b|\bsince\b|\bbecause\b|\bimplies?\b',r'\bhowever\b|\bbut\b|\balthough\b']},
 'criterion_definition_rationale':{'seed_operation':'separate_criterion_from_definition',
  'description':'Expose the introduction of an existing criterion or definition separately from its substantive content, then attach only a source-supported rationale if present. The validated seed contrasts advance identification with retrospective assessment. Other definitions are related variants, not exact replications.',
  'roles':['context','introduction','definition','rationale','qualification','example'],
  'required_roles':['introduction','definition'],
  'language_controls':['mark_existing_contrast','echo_source_criterion_qualifier','retain_attribution'],
  'cue_patterns':[r'\bcriteria?\b',r'\bdefin\w*\b|\bmeans\b|\bcall(?:ed)?\b',r'\bclassif\w*\b|\bdistinguish\w*\b']},
 'staged_existing_components':{'seed_operation':'stage_existing_decisions',
  'description':'Introduce an existing task/component set, then distinguish its already-supplied decisions or components in order. A decision-policy seed is not equivalent to any numerical list. Other processes/components are related variants. Do not invent a count, component or recommendation.',
  'roles':['scope','recommendation','component_introduction','component','qualification','conclusion'],
  'required_roles':['component_introduction','component'],
  'language_controls':['mark_existing_sequence','retain_deontic_scope','retain_component_count'],
  'cue_patterns':[r'\bfirst\b|\bsecond\b|\bthird\b|\bfinally\b',r'\bcomponents?\b|\bstages?\b|\bsteps?\b|\bfunctions?\b|\bdecisions?\b|\bproblems?\b',r'\b(?:two|three|four)\s+(?:decisions|problems|functions|components|stages|steps)\b']}}
SEED_ROLE_SEQUENCES={
 'premise_inference_scope':[['premise','inference','scope'],['premise','inference','qualification']],
 'criterion_definition_rationale':[['context','introduction','definition','rationale']],
 'staged_existing_components':[['scope','recommendation','component_introduction','component','component']]}
for _family,_sequences in SEED_ROLE_SEQUENCES.items():FAMILIES[_family]['seed_role_sequences']=_sequences

ANCHORS={
 'premise_inference_scope':'F1965::fama_1965_behavior_stock_market_prices_p03_0012',
 'criterion_definition_rationale':'F1965::fama_1965_behavior_stock_market_prices_p08_0047',
 'staged_existing_components':'F1965::fama_1965_behavior_stock_market_prices_p08_0049'}
PROBES=['ME1997::FAMA1997_0025','FF2000::FF00-DD-0070','FJ1983::fama_jensen_1983_soc_M0038']
STATUSES=('supported','already_structured','unsupported','uncertain')
KINDS=('actor','quantity','condition','uncertainty','negation','scope','comparison','attribution','causal_direction')
RELATIONS=('inference','conditional','causal','contrast','alternative','presentational')
ROLES=sorted({v for f in FAMILIES.values() for v in f['roles']})
LANGUAGE=sorted({v for f in FAMILIES.values() for v in f['language_controls']})
PILOT={'status':'design_only_annotation_gate_closed','model_id':'google/gemma-3-4b-it',
 'model_revision':'093f9f388b31de276ce2de164bdc2081324b9767','arms':['direct_shared_evidence','operation_shared_evidence'],
 'epochs':4,'minimum_pairs':20,'maximum_pairs':30,'minimum_papers':3,'minimum_papers_per_family':2,
 'minimum_pairs_per_family':4,'maximum_paper_fraction':0.40,'rank':8,'alpha':16,'dropout':0.05,'learning_rate':5e-5,
 'accumulation':2,'seed':42,'training_max_length_ceiling':2048,'generation_context_ceiling':4096,'no_truncation':True,'training_called':False,'requires_richer_format_at_training_and_evaluation':True,
 'related_variants_require_separate_realization_check':True,'reserved_test_rows_used':0}


def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
def arr(item,minimum=0):return {'type':'array','items':item,'minItems':minimum}
def enum(values):return {'type':'string','enum':list(values)}
EVIDENCE=obj({'quote':{'type':'string','minLength':1},'occurrence':{'type':'integer','minimum':1}})
REFS=arr({'type':'string','pattern':'^c[1-9][0-9]*$'},1)
SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema',**obj({
 'case_id':{'type':'string'},'pair_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'status':enum(STATUSES),
 'family':enum(list(FAMILIES)+['none']),'variant':enum(['seed_exact','related_variant','none']),
 'claims':arr(obj({'claim_id':{'type':'string','pattern':'^c[1-9][0-9]*$'},'epistemic_status':enum(['finding','hypothesis','definition','recommendation','assertion','expectation','unresolved']),'composite':{'type':'boolean'},
  'source_evidence':arr(EVIDENCE,1),'target_evidence':arr(EVIDENCE,1),
  'protected':arr(obj({'kind':enum(KINDS),'source_evidence':arr(EVIDENCE,1),'target_evidence':arr(EVIDENCE,1)}))})),
 'source_units':arr(obj({'unit_id':{'type':'string'},'claim_refs':REFS})),
 'target_units':arr(obj({'unit_id':{'type':'string'},'claim_refs':REFS})),
 'relations':arr(obj({'relation_id':{'type':'string','pattern':'^r[1-9][0-9]*$'},'kind':enum(RELATIONS),
  'from_claim_refs':REFS,'to_claim_refs':REFS,'source_evidence':arr(EVIDENCE,1),'certainty':enum(['explicit','interpretive'])})),
 'program':{'anyOf':[{'type':'null'},obj({'stages':arr(obj({'role':enum(ROLES),'claim_refs':REFS,'target_evidence':arr(EVIDENCE,1)}),2),
  'preserve_relation_refs':arr({'type':'string','pattern':'^r[1-9][0-9]*$'}),
  'language_controls':arr(enum(LANGUAGE))})]},
 'notes':{'type':'string'}})}


def validate_json(value,schema):
    if 'anyOf' in schema:
        for choice in schema['anyOf']:
            try:validate_json(value,choice);return
            except ValueError:pass
        raise ValueError('schema_union_mismatch')
    kind=schema['type']
    if kind=='object':
        if not isinstance(value,dict) or set(value)!=set(schema['properties']):raise ValueError('schema_fields')
        for k,s in schema['properties'].items():validate_json(value[k],s)
    elif kind=='array':
        if not isinstance(value,list) or len(value)<schema['minItems']:raise ValueError('schema_array')
        for v in value:validate_json(v,schema['items'])
    elif kind=='string':
        if not isinstance(value,str) or len(value)<schema.get('minLength',0) or ('enum' in schema and value not in schema['enum']) or ('pattern' in schema and not re.fullmatch(schema['pattern'],value)):raise ValueError('schema_string')
    elif kind=='integer':
        if type(value) is not int or value<schema.get('minimum',0):raise ValueError('schema_integer')
    elif kind=='boolean':
        if type(value) is not bool:raise ValueError('schema_boolean')
    elif kind=='null':
        if value is not None:raise ValueError('schema_null')


def pair_sha(row):return sha(json.dumps([row['plain_english'],row['completion']],ensure_ascii=False,separators=(',',':')))
def cue_score(row,family):return sum(min(3,len(re.findall(p,row['completion'],re.I))) for p in FAMILIES[family]['cue_patterns'])


def candidate_pool(rows):
    pool=[];excluded=Counter()
    for r in rows:
        if r['split']!='train':raise ValueError('Only training rows allowed')
        if not 25<=len(r['plain_english'].split())<=180 or not 25<=len(r['completion'].split())<=220:excluded['word_budget']+=1;continue
        a=grouping.output_units(r['plain_english']);b=grouping.output_units(r['completion'])
        if a['ambiguous'] or b['ambiguous']:excluded['uncertain_unit_partition']+=1;continue
        pool.append(r)
    return pool,dict(excluded)


def select_candidates(rows):
    pool,excluded=candidate_pool(rows);by_id={r['id']:r for r in pool};selected=[];used=set();paper_counts=Counter()
    # Exactly one previously positive anchor per family; cap each paper globally.
    for family,ident in ANCHORS.items():
        if ident not in by_id:raise ValueError('Positive anchor no longer eligible')
        r=by_id[ident];selected.append((r,family,'previously_positive_anchor'));used.add(ident);paper_counts[r['paper_key']]+=1
    for ident in PROBES:
        if ident not in by_id:raise ValueError('Probe no longer eligible')
        r=by_id[ident];selected.append((r,'none','preservation_or_no_increment_probe'));used.add(ident);paper_counts[r['paper_key']]+=1
    family_counts=Counter(f for _,f,_ in selected if f!='none');family_papers={f:Counter(r['paper_key'] for r,x,_ in selected if x==f) for f in FAMILIES}
    # Round-robin cue ranking, diversity before cue strength. This is not eligibility.
    for _ in range(8):
        for family in FAMILIES:
            eligible=[r for r in pool if r['id'] not in used and paper_counts[r['paper_key']]<6 and cue_score(r,family)>0]
            if not eligible:raise ValueError('Insufficient diverse candidates; do not force unsupported labels')
            def rank(r):
                a=grouping.output_units(r['plain_english']);b=grouping.output_units(r['completion'])
                return (family_papers[family][r['paper_key']],paper_counts[r['paper_key']],-cue_score(r,family),
                    -int(len(a['units'])!=len(b['units'])),len(r['plain_english'].split())+len(r['completion'].split()),r['id'])
            r=min(eligible,key=rank);selected.append((r,family,'cue_selected_pending_annotation'));used.add(r['id']);paper_counts[r['paper_key']]+=1;family_papers[family][r['paper_key']]+=1;family_counts[family]+=1
    if len(selected)!=30 or set(family_counts.values())!={9}:raise ValueError('Wrong candidate count')
    return selected,{'pool_rows':len(pool),'exclusion_counts':excluded,'paper_counts':dict(paper_counts),'family_candidate_counts':dict(family_counts),
        'selection':'Fixed three positive anchors + three preservation/no-increment probes + 24 round-robin cue/diversity candidates. Family hints are not annotation labels or eligibility.'}


def task_prompt():
    return '''Annotate every pair in the attached annotation_batch_XX.jsonl, using annotation.schema.json.
This is source/target alignment and rhetorical-program annotation, NOT rewriting.
Both texts are supplied for this annotation stage. Keep them unchanged. Do not
produce replacement author targets, new neutral paraphrases or model outputs.

Use the provided operation_families.json. The three successful seed operations
came from one paper. A similar definition/process in another paper is only a
related_variant unless it satisfies the exact seed operation. Never force a
family onto a paragraph to reach a quota. Use already_structured if no meaningful
organization change is demonstrated, unsupported if none applies, and uncertain
for unresolved claim alignment, extraction or semantic discrepancies. Explain why.

For supported/already_structured pairs, inventory substantive claims with local
c1,c2,... IDs and source-based epistemic status (finding, hypothesis, definition,
recommendation, assertion, expectation or unresolved). Bind each to exact source AND target evidence using quote plus a
one-based occurrence. Mark composite claims; do not pretend they are atomic.
Protect actors, quantities and their roles/periods, conditions, modality, scope,
negation, attribution, comparisons and causal direction with paired evidence.
Every provided source and target unit must appear exactly once in its original
order, with the claims it expresses. Do not change unit boundaries or assume
matching wording establishes equivalence. Do not omit supporting claims.

Relations must have source evidence and explicit/interpretive certainty.
Distinguish inference, condition, causation, contrast, alternatives and mere
presentation. Do not introduce a relation just to satisfy a target-style plan.
Build a finite program whose stages reference these SAME source claim IDs and
whose target evidence shows the target actually realizes the move. Use only the
allowed family roles and language controls. Each preserved relation ID must
already exist in your source relation list. Do not invent a component count,
decision, causal mechanism, retrospective criterion, qualification or conclusion.
For a criterion/definition, an introductory phrase is not automatically a separate
substantive claim; explain composite structure instead of fabricating propositions.

Seed-exact means the validated seed's rhetorical action, not merely a keyword.
For seed_exact, use one of the catalog's seed_role_sequences exactly, only when
source and target evidence support those moves. Related variants must be explicitly
marked and need their own realization check.

Target/source scope or strength differences belong in notes; do not silently fix
them or certify exact semantic equivalence. Unsupported/uncertain cases may have
empty inventories and null program; they must include a specific reason.

Return a downloadable UTF-8 annotations_XX.jsonl (same XX as input), one JSON
object per line with exactly the schema fields, copying case_id/pair_sha256.
No Markdown fences, grades, rewritten paragraphs or model-version guesses.
Keep the first response. If incomplete, deliver completed rows and list missing
IDs separately rather than fabricating them. Your annotation is interpretive,
not independent verification. No human-review task is required.
'''


def prepare(output):
    train=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(train)!=TRAIN_SHA:raise ValueError('Frozen training split changed')
    rows=read_lines(train)
    if len(rows)!=247:raise ValueError('Expected 247 training pairs')
    selected,selection=select_candidates(rows);fresh(output);(output/'batches').mkdir();private=[];public=[]
    for row,family,role in selected:
        case='X'+sha(PROTOCOL+'|'+row['id'])[:16]
        record={'case_id':case,'pair_sha256':pair_sha(row),'source':row['plain_english'],'reference':row['completion'],
            'source_units':[{'unit_id':'S'+str(i),'text':t} for i,t in enumerate(grouping.output_units(row['plain_english'])['units'],1)],
            'target_units':[{'unit_id':'T'+str(i),'text':t} for i,t in enumerate(grouping.output_units(row['completion'])['units'],1)]}
        public.append(record);private.append({'case_id':case,'source_id':row['id'],'paper_key':row['paper_key'],'split':'train',
            'candidate_family_hint':family,'selection_role':role,'cue_scores':{f:cue_score(row,f) for f in FAMILIES},'record':record})
    # Mix family hints and probes, same five-record batches; never expose hints.
    public=sorted(public,key=lambda r:sha(r['case_id']))
    jobs={}
    for i in range(6):
        name=f'annotation_batch_{i+1:02d}.jsonl';write_lines(output/'batches'/name,public[i*5:i*5+5])
        for r in public[i*5:i*5+5]:jobs[r['case_id']]=f'annotations_{i+1:02d}.jsonl'
    write_json(output/'private_key.private.json',{'protocol':PROTOCOL,'train_sha256':TRAIN_SHA,'records':private,'response_files':jobs,'selection':selection})
    write_json(output/'annotation.schema.json',SCHEMA);write_json(output/'operation_families.json',{'version':FAMILY_VERSION,'families':FAMILIES,
        'status':'Candidate extensions of validated seed programs; related variants are not already validated.'})
    (output/'ANNOTATION_PROMPT.md').write_text(task_prompt(),encoding='utf-8')
    (output/'START_HERE.md').write_text('''# Operation-family expansion annotation

Thirty candidate pairs, not thirty eligible training examples. Three are positive
anchors, three are preservation/no-increment probes and 24 are new cue-selected
candidates. Keep the original texts unchanged. Family hints are private.

For each XX from 01 to 06, use a separate conversation with the same selected
model (GPT-6 was used for the preceding check). Attach ONLY:
- batches/annotation_batch_XX.jsonl
- annotation.schema.json
- operation_families.json
Paste ANNOTATION_PROMPT.md. Save annotations_XX.jsonl unchanged in:
research_runs/operation_expansion_v1/responses/

Do not attach private_key.private.json, manifest.json, reports or old outputs.
Unlike generation, this annotation stage intentionally includes both source and
author reference. Keep all paragraph files private. Do not supply credentials.
New conversations do not guarantee independence or absence of memory effects.
Do not require a particular family/status or edit the returned annotations.

The CPU auditor will verify hashes, IDs, evidence, claim coverage, source edges
and target move anchors. Related variants remain outside the primary pilot until
a separate realization check. No original source/target is repaired or replaced.
A pilot needs 20-30 supported pairs, at least three papers, at least four pairs
and two papers per family, and no paper above 40 percent. These gates may fail;
if so, report the gap rather than calling the candidates training-ready.

No GPU training or notebook submission is performed by this package. A matched
pilot design is supplied, but no runnable training notebook is released before
the annotation and development-oracle preparation gates are satisfied.
''',encoding='utf-8')
    write_json(output/'pilot_design.json',PILOT)
    write_json(output/'selection_summary.json',{'status':'30_candidates_pending_annotation','candidate_pairs':30,'eligible_pairs':0,'selection':selection,
        'targets_modified':0,'sources_modified':0,'validation_rows_read':0,'reserved_test_rows_read':0,'training_called':False})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),'train_sha256':TRAIN_SHA,
        'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Prepared 30 training-only candidates in six batches. No eligible labels or GPU training claimed.')


def check_package(folder):
    m=json.loads((folder/'manifest.json').read_text())
    if m['protocol']!=PROTOCOL or m['code_sha256']!=file_sha(Path(__file__)):raise ValueError('Compiler/package changed')
    for name,h in m['files'].items():
        if file_sha(folder/name)!=h:raise ValueError('Prepared file changed')
    train=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(train)!=TRAIN_SHA:raise ValueError('Training file changed')
    key=json.loads((folder/'private_key.private.json').read_text());original={r['id']:r for r in read_lines(train)};seen=set()
    for batch in sorted((folder/'batches').glob('*.jsonl')):
        records=read_lines(batch)
        if len(records)!=5:raise ValueError('Wrong batch size')
        for r in records:
            internal=next((a for a in key['records'] if a['case_id']==r['case_id']),None)
            if not internal or r!=internal['record'] or r['case_id'] in seen:raise ValueError('Changed/duplicate record')
            row=original[internal['source_id']]
            if r['source']!=row['plain_english'] or r['reference']!=row['completion'] or r['pair_sha256']!=pair_sha(row):raise ValueError('Original source/target changed')
            seen.add(r['case_id'])
    if len(seen)!=30:raise ValueError('Missing candidates')
    return key


def resolve(text,items):return [grounded.span(text,i['quote'],i['occurrence']) for i in items]
def overlaps(a,b):return a['start']<b['end'] and b['start']<a['end']


def validate_annotation(record,annotation):
    validate_json(annotation,SCHEMA)
    if annotation['case_id']!=record['case_id'] or annotation['pair_sha256']!=record['pair_sha256']:raise ValueError('pair_identity')
    full=annotation['status'] in ('supported','already_structured')
    claims=annotation['claims'];known={c['claim_id'] for c in claims}
    if len(known)!=len(claims):raise ValueError('duplicate_claim_id')
    compiled_claims=[]
    for claim in claims:
        compiled_claims.append({**claim,'source_spans':resolve(record['source'],claim['source_evidence']),
            'target_spans':resolve(record['reference'],claim['target_evidence'])})
        for item in claim['protected']:
            resolve(record['source'],item['source_evidence']);resolve(record['reference'],item['target_evidence'])
    if full and not known:raise ValueError('missing_claims')
    if full:
        for side,text,field in [('source',record['source'],'source_spans'),('target',record['reference'],'target_spans')]:
            expected=record[side+'_units'];assigned=annotation[side+'_units']
            if [r['unit_id'] for r in expected]!=[r['unit_id'] for r in assigned]:raise ValueError('unit_coverage_or_order')
            if set(c for u in assigned for c in u['claim_refs'])!=known:raise ValueError('claim_coverage')
            start=0
            for unit,binding in zip(expected,assigned):
                position=text.find(unit['text'],start)
                if position<0:raise ValueError('input_unit_not_found')
                bounds={'start':position,'end':position+len(unit['text'])};start=bounds['end']
                for ref in binding['claim_refs']:
                    c=next(c for c in compiled_claims if c['claim_id']==ref)
                    if not any(overlaps(s,bounds) for s in c[field]):raise ValueError('unbound_unit_claim')
    relations={r['relation_id']:r for r in annotation['relations']}
    if len(relations)!=len(annotation['relations']):raise ValueError('duplicate_relation_id')
    for relation in relations.values():
        if not set(relation['from_claim_refs']+relation['to_claim_refs'])<=known:raise ValueError('unknown_relation_claim')
        resolve(record['source'],relation['source_evidence'])
    program=annotation['program']
    if annotation['status']=='supported' and (annotation['family']=='none' or program is None or annotation['variant']=='none'):raise ValueError('missing_supported_program')
    if annotation['status'] in ('uncertain','unsupported') and not annotation['notes'].strip():raise ValueError('missing_uncertainty_reason')
    if program:
        if annotation['family'] not in FAMILIES:raise ValueError('program_without_family')
        family=FAMILIES[annotation['family']];roles=[s['role'] for s in program['stages']]
        if not set(roles)<=set(family['roles']) or not set(family['required_roles'])<=set(roles):raise ValueError('wrong_family_roles')
        if annotation['variant']=='seed_exact' and roles not in family['seed_role_sequences']:raise ValueError('not_exact_seed_role_structure')
        if not set(program['language_controls'])<=set(family['language_controls']):raise ValueError('wrong_family_language')
        if not set(program['preserve_relation_refs'])<=set(relations):raise ValueError('invented_relation')
        stage_claims=set();stage_positions=[]
        for stage in program['stages']:
            refs=set(stage['claim_refs']);stage_claims|=refs
            if not refs<=known:raise ValueError('invented_stage_claim')
            evidence=resolve(record['reference'],stage['target_evidence']);stage_positions.append(min(e['start'] for e in evidence))
            for ref in refs:
                c=next(c for c in compiled_claims if c['claim_id']==ref)
                if not any(overlaps(s,t) for s in evidence for t in c['target_spans']):raise ValueError('target_stage_not_anchored')
        if stage_positions!=sorted(stage_positions):raise ValueError('target_stage_order')
        if stage_claims!=known:raise ValueError('program_omits_claims')
        if annotation['family']=='premise_inference_scope':
            premise={c for s in program['stages'] if s['role']=='premise' for c in s['claim_refs']};inference={c for s in program['stages'] if s['role']=='inference' for c in s['claim_refs']}
            if not any(set(relations[i]['from_claim_refs'])<=premise and set(relations[i]['to_claim_refs'])<=inference and relations[i]['kind'] in ('inference','conditional','causal') for i in program['preserve_relation_refs']):raise ValueError('unsupported_premise_inference_link')
            if annotation['variant']=='seed_exact' and not set(roles)&{'scope','qualification'}:raise ValueError('seed_missing_qualified_scope')
    before=[u['claim_refs'] for u in annotation['source_units']];after=[u['claim_refs'] for u in annotation['target_units']]
    def order(field):return [c['claim_id'] for c in sorted(compiled_claims,key=lambda c:min(s['start'] for s in c[field]))]
    contrast=full and (before!=after or order('source_spans')!=order('target_spans'))
    return {'annotation':annotation,'compiled_claims':compiled_claims,'packing_or_order_contrast':contrast,
        'semantic_equivalence_verified':False,'atomic_claim_completeness_verified':False}


def render_pair(record,compiled):
    a=compiled['annotation'];family=FAMILIES[a['family']];source=record['source']
    index=[{'claim_id':c['claim_id'],'source_spans':resolve(source,c['source_evidence']),
        'epistemic_status':c['epistemic_status'],'protected_evidence':[{'kind':p['kind'],'source_spans':resolve(source,p['source_evidence'])} for p in c['protected']]} for c in a['claims']]
    relations=[{'relation_id':r['relation_id'],'kind':r['kind'],'source_claim_refs':r['from_claim_refs'],
        'target_claim_refs':r['to_claim_refs'],'direction':'source_to_target','explicitness':r['certainty'],
        'source_spans':resolve(source,r['source_evidence'])} for r in a['relations'] if r['relation_id'] in a['program']['preserve_relation_refs']]
    payload={'source':source,'source_claim_index':index,'source_facts':[],'source_relations':relations}
    prefix=transport.parent.source_prefix(payload)
    if a['variant']=='seed_exact':
        operation=family['seed_operation'];catalog=grounded.CATALOG[operation]
        program={'version':grounded.VERSION,'operation':operation,
            'stages':[{'role':role,'source_claim_refs':stage['claim_refs']} for role,stage in zip(catalog['roles'],a['program']['stages'])],
            'preserve_source_relation_refs':a['program']['preserve_relation_refs'],'language_actions':catalog['language_actions']}
        guide='\n\nRhetorical program:\n'+json.dumps(program,sort_keys=True)+'\nParagraph action: '+catalog['paragraph_action']+'\nSentence action: '+catalog['sentence_action']+(
            '\nStages are discourse moves, not compulsory one-sentence slots. Preserve source-supported relation direction and scope.')
    else:
        program={'version':FAMILY_VERSION,'family':a['family'],'variant':a['variant'],
            'stages':[{'role':s['role'],'source_claim_refs':s['claim_refs']} for s in a['program']['stages']],
            'preserve_source_relation_refs':a['program']['preserve_relation_refs'],'language_actions':a['program']['language_controls']}
        guide='\n\nRelated-variant hypothesis (not validated for primary training):\n'+json.dumps(program,sort_keys=True)+'\n'+family['description']
    return {'direct_shared_evidence':prefix+'\nReturn only the rewritten paragraph.',
        'operation_shared_evidence':prefix+guide+'\nReturn only the rewritten paragraph.'}


def pilot_gates(eligible,variant_rows):
    counts=Counter(r['paper_key'] for r in eligible);families={f:[r for r in eligible if r['family']==f] for f in FAMILIES}
    checks={'pair_count':20<=len(eligible)<=30,'paper_count':len(counts)>=3,
        'paper_balance':bool(eligible) and max(counts.values())/len(eligible)<=PILOT['maximum_paper_fraction'],
        'family_count':all(len(r)>=4 for r in families.values()),'family_paper_diversity':all(len({p['paper_key'] for p in r})>=2 for r in families.values())}
    return {'training_annotation_gate_passed':all(checks.values()),'checks':checks,'primary_seed_exact_pairs':len(eligible),
        'related_variants_needing_realization_check':len(variant_rows),'paper_counts':dict(counts),
        'family_counts':{f:len(r) for f,r in families.items()},'ready_to_train':False,
        'remaining_gates':['Automated semantic/style inspection of returned alignments and evidence.','Separate oracle development-plan preparation; keep reserved test unused.','Related-variant realization validation before adding those variants.'],
        'training_called':False}


def audit(package,responses,output):
    key=check_package(package);expected={r['case_id']:r for r in key['records']};valid=[];errors=[];observed=[];hashes={}
    for name in sorted(set(key['response_files'].values())):
        path=responses/name
        if not path.exists():errors.append({'file':name,'code':'missing_file'});continue
        hashes[name]=file_sha(path)
        try:lines=path.read_text(encoding='utf-8').splitlines()
        except UnicodeError:errors.append({'file':name,'code':'invalid_utf8'});continue
        for number,line in enumerate(lines,1):
            if not line.strip():continue
            try:
                a=transport.loads_unique(line)
                if isinstance(a,dict) and isinstance(a.get('case_id'),str) and a['case_id'] in expected:observed.append(a['case_id'])
                validate_json(a,SCHEMA);info=expected.get(a['case_id'])
                if not info or key['response_files'][a['case_id']]!=name:raise ValueError('unknown_case_or_wrong_batch')
                compiled=validate_annotation(info['record'],a);valid.append({'case_id':a['case_id'],'provenance':{'file':name,'line':number},**compiled})
            except (ValueError,TypeError,KeyError) as exc:
                code=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[a-z_]+',str(exc)) else 'invalid_annotation'
                errors.append({'file':name,'line':number,'code':code})
    duplicates={c for c,n in Counter(observed).items() if n>1}
    valid=[v for v in valid if v['case_id'] not in duplicates]
    errors.extend({'case_id':c,'code':'duplicate_all_copies_excluded'} for c in sorted(duplicates))
    eligible=[];variants=[];dispositions=[]
    for c in valid:
        info=expected[c['case_id']];a=c['annotation'];reason='not_supported'
        if info['selection_role']=='preservation_or_no_increment_probe':reason='reserved_diagnostic_probe'
        elif a['status']=='supported' and not c['packing_or_order_contrast']:reason='no_packing_or_order_contrast'
        elif a['status']=='supported':
            item={'case_id':c['case_id'],'source_id':info['source_id'],'paper_key':info['paper_key'],'family':a['family'],'variant':a['variant'],
                'source':info['record']['source'],'reference':info['record']['reference'],'source_sha256':sha(info['record']['source']),
                'reference_sha256':sha(info['record']['reference']),'prompts':render_pair(info['record'],c),'annotation_sha256':sha(json.dumps(a,sort_keys=True,ensure_ascii=False)),
                'semantic_equivalence_verified':False}
            if a['variant']=='seed_exact':eligible.append(item);reason='provisional_seed_exact_pair'
            else:variants.append(item);reason='related_variant_requires_realization_check'
        dispositions.append({'case_id':c['case_id'],'status':a['status'],'family':a['family'],'reason':reason})
    fresh(output);(output/'raw').mkdir()
    for name in hashes:shutil.copyfile(responses/name,output/'raw'/name)
    write_lines(output/'annotations.checked.private.jsonl',valid);write_lines(output/'primary_pairs.provisional.private.jsonl',eligible);write_lines(output/'related_variants.private.jsonl',variants)
    write_json(output/'dispositions.json',dispositions);write_json(output/'pilot_gate.json',pilot_gates(eligible,variants))
    write_json(output/'coverage.json',{'expected':30,'accepted_annotations':len(valid),'missing_or_rejected':sorted(set(expected)-{r['case_id'] for r in valid}),'errors':errors,
        'complete':len(valid)==30 and not errors,'response_hashes':hashes,'training_called':False})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'package_manifest_sha256':file_sha(package/'manifest.json'),'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Annotation audit saved with explicit eligibility gaps. No training or model calls executed.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--check-package',action='store_true');a.add_argument('--audit',action='store_true')
    p.add_argument('--package',type=Path);p.add_argument('--responses-dir',type=Path);p.add_argument('--output-dir',type=Path);args=p.parse_args(argv)
    if args.prepare:
        if args.output_dir is None:p.error('--output-dir required')
        prepare(args.output_dir)
    elif args.check_package:
        if args.package is None:p.error('--package required')
        check_package(args.package);print('Thirty frozen training candidates verified; no model calls.')
    else:
        if args.package is None or args.responses_dir is None or args.output_dir is None:p.error('--package, --responses-dir and --output-dir required')
        audit(args.package,args.responses_dir,args.output_dir)


if __name__=='__main__':main()
