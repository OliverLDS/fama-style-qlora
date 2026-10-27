"""CPU-only, claim-bound rhetorical programs from frozen training annotations.

This is an evidence compiler for five declared cases, not an automatic semantic
parser, generator, learned mapper or validation of author-specific style.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from .concise_plan import TRAIN_SHA,file_sha,read_lines
from .experiment import fresh,write_json,write_lines
from fama_style.research.claims import sha

ROOT=Path(__file__).resolve().parents[3]
VERSION='grounded_rhetorical_operations_v1'
CATALOG={
 'separate_premise_inference_qualification':{
  'paragraph_action':'Present an existing premise, its already-supported inference, and the qualified scope of practical applicability as distinct moves.',
  'sentence_action':'Separate the premise from its inference; keep each condition attached to the applicability claim it qualifies.',
  'language_actions':['retain_modal_strength','mark_existing_scope_contrast'],
  'roles':['premise','supported_inference','conditional_applicability']},
 'separate_criterion_from_definition':{
  'paragraph_action':'Contrast advance identification with a retrospective criterion, define the criterion, then justify its persistence requirement using the supplied counterexample.',
  'sentence_action':'Introduce the criterion separately from its definition; group the criterion emphasis with its existing rationale.',
  'language_actions':['mark_existing_contrast','echo_source_criterion_qualifier'],
  'roles':['advance_identification','criterion_introduction','criterion_definition','criterion_rationale']},
 'stage_existing_decisions':{
  'paragraph_action':'Retain the scope and recommendation, introduce only the source-stated component count, then develop the existing decisions in order.',
  'sentence_action':'Separate the count announcement from the decisions; distinguish the first decision and coordinate the remaining decisions without adding steps.',
  'language_actions':['mark_existing_sequence','retain_deontic_scope'],
  'roles':['scope','recommendation','component_count','first_decision','remaining_decisions']},
 'bind_comparison_to_supported_inference':{
  'paragraph_action':'Distinguish composition from propensity, present the characteristic component, attach the controlled inference to its comparison, and retain period and single-year evidence.',
  'sentence_action':'Integrate an existing expected/actual comparison with its existing inference, without changing which quantities are expected, actual or differences.',
  'language_actions':['mark_existing_comparison','retain_evidence_strength'],
  'roles':['cohort_and_components','characteristic_component','controlled_inference','period_evidence','single_year_evidence_and_inference']},
 'retain_parallel_alternatives':{
  'paragraph_action':'Retain parallel finding/alternative-explanation blocks; do not choose among the alternatives.',
  'sentence_action':'Keep alternatives paired with their corresponding finding and preserve their hypothetical status.',
  'language_actions':['retain_hypothetical_status','retain_attribution_scope'],
  'roles':['classification_ambiguity','first_finding','first_alternatives','second_finding','second_alternatives']}
}
# Bindings below contain IDs and small evidence cues only, never paragraph text.
# Alignments are declared, inspectable hypotheses; they are not derived by matching
# c-number identities between independently annotated source and target.
SPECS=[
 {'id':'F1965::fama_1965_behavior_stock_market_prices_p03_0012','operation':'separate_premise_inference_qualification',
  'alignment':{'c1':['c1'],'c2':['c2'],'c3':['c3']},'target_groups':[['c1'],['c2'],['c3']],
  'relation_refs':['r1'],'target_cues':['Thus','For practical purposes','as long as'],
  'language_labels':['lexical_or_modal_hedge'],'caveat':'Source unlikely versus target probably never hope differs in force. Preserve source strength rather than imitating this difference.'},
 {'id':'F1965::fama_1965_behavior_stock_market_prices_p08_0047','operation':'separate_criterion_from_definition',
  'alignment':{'c1':['c1'],'c2':['c2'],'c3':['c2'],'c4':['c3'],'c5':['c3'],'c6':['c3']},
  'target_groups':[['c1'],['c2'],['c3'],['c4','c5','c6']],
  'relation_refs':['r1'],'target_cues':['Ex post','Consistently','since'],
  'language_labels':['contrastive_connective','conditional_subordinator'],
  'caveat':'Source c2 and c3 are composite claims. Splitting their exposition does not add new independent propositions.'},
 {'id':'F1965::fama_1965_behavior_stock_market_prices_p08_0049','operation':'stage_existing_decisions',
  'alignment':{'c1':['c1'],'c2':['c2'],'c3':['c3','c4','c5'],'c4':['c3'],'c5':['c4'],'c6':['c5']},
  'target_groups':[['c1'],['c2'],['c3'],['c4'],['c5','c6']],
  'relation_refs':['r1'],'target_cues':['three separate problems','First','Then','finally'],
  'language_labels':['additive_or_sequential_connective'],
  'source_count_anchor':'This involves three decisions','source_count':3,
  'caveat':'The source annotation omits a separate claim for its explicit component-count introduction. Record this source span separately; do not silently repair the old annotation. Target at every point in time is stronger than source current prices.'},
 {'id':'FF2000::FF00-DD-0070','operation':'bind_comparison_to_supported_inference',
  'alignment':{'c1':['c1'],'c2':['c2'],'c3':['c3'],'c4':['c4'],'c5':['c5'],'c6':['c6'],'c7':['c6'],'c8':['c7']},
  'target_groups':[['c1'],['c2'],['c3','c4'],['c5'],['c6','c7','c8']],
  'relation_refs':['r1','r3'],'target_cues':['controlling for characteristics','expected versus','rather strong evidence'],
  'language_labels':['comparative_marker','compressed_multicomponent_expression'],
  'quantity_roles':[
   ['predicted_rate','start_year','c2','c2','11.3'],['predicted_rate','end_year','c2','c2','5.2'],
   ['difference','period_average','c5','c5','3.8'],['predicted_rate','period_average','c5','c5','6.8'],
   ['actual_rate','period_average','c5','c5','3.0'],['predicted_rate','single_year','c6','c6','5.2'],
   ['actual_rate','single_year','c6','c6','0.8']],
  'caveat':'The source annotation additionally links period-average evidence to the final inference; that edge is retained in the ledger but is not introduced or promoted by this program. Numeric-role labels are declared and span-bound, not an independent economic parse.'},
 {'id':'ME1997::FAMA1997_0025','operation':'retain_parallel_alternatives',
  'alignment':{'c'+str(i):['c'+str(i)] for i in range(1,8)},
  'target_groups':[['c1'],['c2'],['c3','c4'],['c5'],['c6','c7']],
  'relation_refs':['r1','r2','r3','r4'],'target_cues':['might be attributed','One story','another'],
  'language_labels':['lexical_or_modal_hedge'],
  'caveat':'Source and reference already have the same claim packing. This is a preservation control, not a demonstrated structural transformation; different annotation names alone do not establish contrast.'}
]


RELATION_ENDPOINTS={
 'separate_premise_inference_qualification':{'r1':[['c1'],['c2']]},
 'separate_criterion_from_definition':{'r1':[['c3'],['c2']]},
 'stage_existing_decisions':{'r1':[['c1'],['c2']]},
 'bind_comparison_to_supported_inference':{'r1':[['c3'],['c4']],'r3':[['c6'],['c7']]},
 'retain_parallel_alternatives':{'r1':[['c2'],['c3']],'r2':[['c2'],['c4']],'r3':[['c5'],['c6']],'r4':[['c5'],['c7']]}}


def check_relations(relations,operation):
    indexed={r['relation_id']:r for r in relations}
    for key,ends in RELATION_ENDPOINTS[operation].items():
        r=indexed.get(key)
        if not r or r['direction']!='source_to_target' or [r['source_claim_refs'],r['target_claim_refs']]!=ends:raise ValueError('Missing, reversed or rebound source relation')


def program_json_schema():
    branches=[]
    def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    def ids(pattern):return {'type':'array','items':{'type':'string','pattern':pattern},'minItems':1,'uniqueItems':True}
    for name,catalog in CATALOG.items():
        stages=[]
        for role in catalog['roles']:
            fields={'role':{'const':role},'source_claim_refs':ids('^c[1-9][0-9]*$')}
            if role=='component_count':fields['source_fact_refs']={'const':['f1']}
            stages.append(obj(fields))
        branches.append(obj({'version':{'const':VERSION},'operation':{'const':name},
            'stages':{'type':'array','prefixItems':stages,'minItems':len(stages),'maxItems':len(stages)},
            'preserve_source_relation_refs':ids('^r[1-9][0-9]*$'),
            'language_actions':{'const':catalog['language_actions']}}))
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:fama-style:grounded-rhetorical-operations:v1','oneOf':branches}


def span(text,quote,occurrence=1):
    if not isinstance(quote,str) or not quote or type(occurrence) is not int or occurrence<1:raise ValueError('Unresolved evidence anchor')
    matches=list(re.finditer(re.escape(quote),text))
    if len(matches)<occurrence:raise ValueError('Evidence missing from exact text')
    match=matches[occurrence-1]
    return {'start':match.start(),'end':match.end(),'quote':quote}


def evidence_spans(text,items):
    return [span(text,e['quote'],e.get('occurrence',1)) for e in items]


def annotation_view(text,annotation):
    semantic=annotation['semantic_content'];claims=semantic['claims'];entities=semantic['entities']
    if len({c['claim_id'] for c in claims})!=len(claims):raise ValueError('Duplicate claim IDs')
    known={c['claim_id'] for c in claims};entity_ids={e['entity_id'] for e in entities}
    if any(not set(c['entity_refs'])<=entity_ids for c in claims):raise ValueError('Unknown actor/entity reference')
    for c in claims:
        if not c['evidence']:raise ValueError('Claim without source evidence')
        evidence_spans(text,c['evidence'])
    units=annotation['rhetorical_taxonomy']['sentence_structure']
    if ' '.join(' '.join(u['sentence_text'].split()) for u in units)!=' '.join(text.split()):raise ValueError('Annotation text coverage mismatch')
    if set(c for u in units for c in u['claim_refs'])!=known:raise ValueError('Missing or invented sentence claims')
    for r in semantic['relations']:
        if not set(r['source_claim_refs']+r['target_claim_refs'])<=known:raise ValueError('Unknown semantic relation endpoint')
    return {'semantic_content':copy.deepcopy(semantic),
        'claim_spans':{c['claim_id']:evidence_spans(text,c['evidence']) for c in claims},
        'sentence_claim_groups':[u['claim_refs'] for u in units],
        'paragraph_moves':[{'role':m['role']['name'],'claim_refs':m['claim_refs']} for m in annotation['rhetorical_taxonomy']['paragraph_structure']['moves']],
        'sentence_labels':sorted({f['label']['name'] for u in units for f in u['features']}),
        'language_labels':sorted({f['label']['name'] for f in annotation['rhetorical_taxonomy']['language_features']}),
        'annotation_sha256':sha(json.dumps(annotation,sort_keys=True,ensure_ascii=False))}


def unique(items):return list(dict.fromkeys(items))


def compile_card(row,source_annotation,target_annotation,spec):
    if row['split']!='train' or row['id']!=spec['id']:raise ValueError('Only declared training examples allowed')
    source=row['plain_english'];target=row['completion']
    a=annotation_view(source,source_annotation);b=annotation_view(target,target_annotation)
    source_ids=set(a['claim_spans']);target_ids=set(b['claim_spans']);mapping=spec['alignment']
    if set(mapping)!=target_ids or set(c for ids in mapping.values() for c in ids)!=source_ids:raise ValueError('Incomplete or invented alignment')
    if b['sentence_claim_groups']!=spec['target_groups']:raise ValueError('Reference does not realize declared packing')
    roles=CATALOG[spec['operation']]['roles']
    if len(roles)!=len(spec['target_groups']):raise ValueError('Stage mismatch')
    source_relations={r['relation_id']:r for r in a['semantic_content']['relations']}
    if not set(spec['relation_refs'])<=set(source_relations):raise ValueError('Unsupported semantic relation')
    check_relations(a['semantic_content']['relations'],spec['operation'])
    if not set(spec['language_labels'])<=set(b['language_labels']):raise ValueError('Missing target language-feature evidence')
    stages=[]
    for role,group in zip(roles,spec['target_groups']):
        refs=unique(c for t in group for c in mapping[t])
        stages.append({'role':role,'source_claim_refs':refs})
    extra=[]
    if 'source_count_anchor' in spec:
        extra=[{'fact_id':'f1','kind':'existing_component_count','value':spec['source_count'],
            'source_span':span(source,spec['source_count_anchor'])}]
        if spec['source_count']!=len(stages[2]['source_claim_refs']) or 'three' not in extra[0]['source_span']['quote'].lower():raise ValueError('Unsupported component count')
        stages[2]['source_fact_refs']=['f1']
    quantities=[]
    for role,time_scope,sc,tc,value in spec.get('quantity_roles',[]):
        def local_numeric(claim_spans,text):
            for bounds in claim_spans:
                m=re.search(r'(?<![\d.])'+re.escape(value)+r'(?![\d.])',bounds['quote'])
                if m:return {'quote':value,'start':bounds['start']+m.start(),'end':bounds['start']+m.end()}
            raise ValueError('Quantity missing from bound claim')
        quantities.append({'role':role,'time_scope':time_scope,'source_claim_ref':sc,'target_claim_ref':tc,'value':value,
            'source_span':local_numeric(a['claim_spans'][sc],source),'target_span':local_numeric(b['claim_spans'][tc],target)})
    projected=[s['source_claim_refs'] for s in stages];contrast=projected!=a['sentence_claim_groups']
    if (spec['operation']=='retain_parallel_alternatives')==contrast:raise ValueError('Control/contrast assignment not demonstrated')
    return {'version':VERSION,'id':row['id'],'paper_key':row['paper_key'],'split':'train',
        'source':source,'reference':target,'source_sha256':sha(source),'reference_sha256':sha(target),
        'source_annotation':a,'target_annotation':b,'target_to_source_claim_alignment':copy.deepcopy(mapping),
        'program':{'version':VERSION,'operation':spec['operation'],'stages':stages,
            'preserve_source_relation_refs':spec['relation_refs'],'language_actions':CATALOG[spec['operation']]['language_actions']},
        'source_fact_supplements':extra,'quantity_role_bindings':quantities,
        'target_realization_checks':{'declared_claim_packing_matches_annotation':True,'ordered_target_cue_spans':[span(target,q) for q in spec['target_cues']],
            'language_labels_present':spec['language_labels'],'source_packing_differs':contrast,
            'semantic_equivalence_proved':False,'author_specificity_proved':False},
        'use':'contrast_hypothesis' if contrast else 'preservation_control',
        'uncertainty':spec['caveat'],'training_ready':False,
        'provenance':'Fixed, in-session claim alignment over previously checked independent LLM source/target annotations. Span, ID, packing and cue checks are executable; semantic alignments and rhetorical readings remain interpretations.'}


def validate_card(card):
    spec=next((s for s in SPECS if s['id']==card['id']),None)
    if not spec:raise ValueError('Unknown operation example')
    # Rebuild from the immutable upstream inputs when validating a released pack;
    # this local contract also rejects invalid/changed plan references immediately.
    program=card['program'];catalog=CATALOG.get(program.get('operation'))
    if set(program)!={'version','operation','stages','preserve_source_relation_refs','language_actions'} or not catalog or program['version']!=VERSION:raise ValueError('Invalid closed program')
    if program['operation']!=spec['operation'] or program['language_actions']!=catalog['language_actions']:raise ValueError('Changed operation or language action')
    if len(program['stages'])!=len(catalog['roles']):raise ValueError('Wrong stage count')
    for i,stage in enumerate(program['stages']):
        allowed={'role','source_claim_refs'}|({'source_fact_refs'} if spec.get('source_count_anchor') and i==2 else set())
        expected=unique(c for t in spec['target_groups'][i] for c in spec['alignment'][t])
        if set(stage)!=allowed or stage['role']!=catalog['roles'][i] or stage['source_claim_refs']!=expected:raise ValueError('Changed stage/claim binding')
        if 'source_fact_refs' in stage and stage['source_fact_refs']!=['f1']:raise ValueError('Invented source fact')
    if program['preserve_source_relation_refs']!=spec['relation_refs']:raise ValueError('Invented or changed semantic relation')
    if card['source_sha256']!=sha(card['source']) or card['reference_sha256']!=sha(card['reference']):raise ValueError('Text identity mismatch')
    def check_span(text,bounds):
        if type(bounds.get('start')) is not int or type(bounds.get('end')) is not int or not 0<=bounds['start']<bounds['end']<=len(text) or text[bounds['start']:bounds['end']]!=bounds['quote']:raise ValueError('Changed evidence span')
    for field,text_field in [('source_annotation','source'),('target_annotation','reference')]:
        view=card[field];text=card[text_field]
        known={c['claim_id'] for c in view['semantic_content']['claims']}
        if known!=set(view['claim_spans']) or set(c for group in view['sentence_claim_groups'] for c in group)!=known:raise ValueError('Changed annotation coverage')
        for bounds in view['claim_spans'].values():
            for b in bounds:check_span(text,b)
    if card['target_to_source_claim_alignment']!=spec['alignment'] or card['target_annotation']['sentence_claim_groups']!=spec['target_groups']:raise ValueError('Changed reference alignment/packing')
    check_relations(card['source_annotation']['semantic_content']['relations'],spec['operation'])
    expected_facts=[]
    if 'source_count_anchor' in spec:
        expected_facts=[{'fact_id':'f1','kind':'existing_component_count','value':spec['source_count'],'source_span':span(card['source'],spec['source_count_anchor'])}]
    if card['source_fact_supplements']!=expected_facts:raise ValueError('Invented or changed supplemental fact')
    realized=card['target_realization_checks']
    if realized['ordered_target_cue_spans']!=[span(card['reference'],q) for q in spec['target_cues']]:raise ValueError('Changed target realization cues')
    positions=[bounds['start'] for bounds in realized['ordered_target_cue_spans']]
    if positions!=sorted(positions):raise ValueError('Target cue progression not realized')
    observed=card['source_annotation']['sentence_claim_groups']!=[stage['source_claim_refs'] for stage in program['stages']]
    if realized['source_packing_differs'] is not observed or card['use']!=('contrast_hypothesis' if observed else 'preservation_control') or card['training_ready'] is not False:raise ValueError('Misrepresented contrast/control or training readiness')
    actual=[[q['role'],q['time_scope'],q['source_claim_ref'],q['target_claim_ref'],q['value']] for q in card['quantity_role_bindings']]
    if actual!=spec.get('quantity_roles',[]):raise ValueError('Changed numeric role bindings')
    for q in card['quantity_role_bindings']:
        for side,annotation,text in [('source','source_annotation',card['source']),('target','target_annotation',card['reference'])]:
            b=q[side+'_span'];check_span(text,b)
            if b['quote']!=q['value'] or not any(a['start']<=b['start']<b['end']<=a['end'] for a in card[annotation]['claim_spans'][q[side+'_claim_ref']]):raise ValueError('Unbound numeric role evidence')
    return card


def compile_program(card):
    """Return only executable source-bound presentation controls, not target data."""
    validate_card(card)
    return copy.deepcopy(card['program'])


def generation_payload(card):
    program=compile_program(card)
    claims=card['source_annotation']['semantic_content']['claims'];source=card['source']
    index=[]
    for c in claims:
        protected=[]
        for item in c['qualifications']:
            protected.append({'kind':item['kind'],'scope_claim_refs':item['scope_claim_refs'],'source_spans':evidence_spans(source,item['evidence'])})
        for item in c['quantities']:
            protected.append({'kind':'quantity','source_spans':evidence_spans(source,item['evidence'])})
        index.append({'claim_id':c['claim_id'],'source_spans':card['source_annotation']['claim_spans'][c['claim_id']],
            'epistemic_status':c['epistemic_status'],'protected_evidence':protected})
    relations=[{k:r[k] for k in ('relation_id','kind','source_claim_refs','target_claim_refs','direction','explicitness')}|
        {'source_spans':evidence_spans(source,r['evidence'])} for r in card['source_annotation']['semantic_content']['relations'] if r['relation_id'] in program['preserve_source_relation_refs']]
    return {'version':VERSION,'source':source,'source_claim_index':index,'source_facts':copy.deepcopy(card['source_fact_supplements']),
        'source_relations':relations,'program':program}


def render_payload(payload):
    if set(payload)!={'version','source','source_claim_index','source_facts','source_relations','program'} or payload['version']!=VERSION:raise ValueError('Unexpected generator payload fields')
    operation=CATALOG[payload['program']['operation']]
    return ('Rewrite the source using its supported rhetorical program. Return one paragraph. '
        'Use the source claims and preserve actors, attributions, numeric roles, conditions, uncertainty, negation and causal direction. '
        'Stages are discourse moves, not compulsory sentence slots. Reorganize presentation without adding relations or calculations. '
        'Do not strengthen a hypothesis into a finding or invent a conclusion.\n\nSource:\n'+payload['source']+
        '\n\nSource claim index (fallible annotations, not additional facts):\n'+json.dumps(payload['source_claim_index'],ensure_ascii=False,sort_keys=True)+
        '\nSource-bound supplemental evidence:\n'+json.dumps(payload['source_facts'],ensure_ascii=False,sort_keys=True)+
        '\nExisting source relation evidence (interpretive):\n'+json.dumps(payload['source_relations'],ensure_ascii=False,sort_keys=True)+
        '\nProgram:\n'+json.dumps(payload['program'],sort_keys=True)+
        '\nParagraph action: '+operation['paragraph_action']+'\nSentence action: '+operation['sentence_action'])


def render_prompt(card):return render_payload(generation_payload(card))


def build_cards():
    root=ROOT/'research_runs/taxonomy_discovery_v1'
    training=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(training)!=TRAIN_SHA:raise ValueError('Frozen training corpus changed')
    paths={'mapping':root/'package_v1/PRIVATE_MAPPING_DO_NOT_UPLOAD.json',
        'target':root/'response_audit_v1/original_annotations.checked.private.jsonl',
        'source':root/'response_audit_v2/annotations_B.normalized.private.jsonl',
        'neutral':root/'response_audit_v1/neutral_proposals.checked.private.jsonl'}
    for path in paths.values():
        m=json.loads((path.parent/'manifest.json').read_text())
        if file_sha(path)!=m['files'][path.name]:raise ValueError('Upstream audited annotation changed')
    inputs={str(p.relative_to(ROOT)):file_sha(p) for p in [training]+list(paths.values())}
    rows={r['id']:r for r in read_lines(training)}
    if len(rows)!=247 or any(r['split']!='train' for r in rows.values()):raise ValueError('Only frozen training split allowed')
    keys={r['training_id']:r for r in json.loads(paths['mapping'].read_text())}
    target={r['paragraph_id']:r for r in read_lines(paths['target'])};source={r['paragraph_id']:r for r in read_lines(paths['source'])};neutral={r['pair_id']:r for r in read_lines(paths['neutral'])}
    result=[]
    for spec in SPECS:
        row=rows[spec['id']];key=keys[spec['id']];proposal=neutral[key['pair_id']]
        if proposal['decision']!='retain' or proposal['proposed_neutral_text']!=row['plain_english']:raise ValueError('Source is an unaccepted revision')
        card=compile_card(row,source[key['reference_id']],target[key['original_item_id']],spec)
        validate_card(card);result.append(card)
    return result,inputs


def prepare(output):
    cards,inputs=build_cards();fresh(output)
    write_lines(output/'operation_cards.private.jsonl',cards)
    write_lines(output/'prompt_preview.private.jsonl',[{'id':c['id'],'use':c['use'],'prompt':render_prompt(c),'prompt_sha256':sha(render_prompt(c))} for c in cards])
    summary={'version':VERSION,'status':'cpu_prepared_not_model_executed','training_rows':len(cards),'training_papers':len({c['paper_key'] for c in cards}),
        'contrast_hypotheses':sum(c['use']=='contrast_hypothesis' for c in cards),'preservation_controls':sum(c['use']=='preservation_control' for c in cards),
        'operation_support_counts':dict(Counter(c['program']['operation'] for c in cards)),
        'numeric_role_bindings':sum(len(c['quantity_role_bindings']) for c in cards),'training_ready':False,
        'model_calls':0,'gpu_jobs_submitted':0,'training_called':False,'validation_rows_read':0,'reserved_test_rows_read':0,
        'limits':'One example per operation; three papers; declared alignments and fallible LLM annotations. Exact spans and reference claim packing are checked, not semantic equivalence or author specificity. A control is not a new training contrast. Programs are reference-informed oracle hypotheses.'}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'version':VERSION,'input_hashes':inputs,'code_sha256':file_sha(Path(__file__).resolve()),'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print('Prepared four grounded contrast hypotheses plus one preservation control. No models, Secrets, GPU or training accessed.')


def check(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['code_sha256']!=file_sha(Path(__file__).resolve()):raise ValueError('Compiler changed')
    for n,h in manifest['files'].items():
        if file_sha(directory/n)!=h:raise ValueError('Artifact changed')
    cards,inputs=build_cards()
    if inputs!=manifest['input_hashes'] or cards!=read_lines(directory/'operation_cards.private.jsonl'):raise ValueError('Cards do not replay from original annotations')
    expected=[{'id':c['id'],'use':c['use'],'prompt':render_prompt(c),'prompt_sha256':sha(render_prompt(c))} for c in cards]
    if expected!=read_lines(directory/'prompt_preview.private.jsonl'):raise ValueError('Prompts do not replay')
    print('All five cards and source-only prompt previews replay from frozen training annotations; no ML or GPU accessed.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--check',action='store_true');args=p.parse_args(argv)
    if args.check:check(args.output_dir)
    else:prepare(args.output_dir)


if __name__=='__main__':main()
