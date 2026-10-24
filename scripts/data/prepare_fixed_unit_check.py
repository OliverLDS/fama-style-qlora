"""Build a two-text, two-round fixed-cell consistency check; no inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[2]
FEATURES=[('F01','P028','paragraph'),('F02','P030','paragraph'),('F03','L006','paragraph'),
          ('F04','L013','paragraph'),('F05','L022','paragraph'),
          ('F06','S020','sentence'),('F07','S024','sentence'),('F08','S025','sentence'),
          ('F09','S031','subsentence'),('F10','S032','subsentence'),('F11','S034','subsentence'),
          ('F12','P071','sentence_pair'),('F13','P072','sentence_pair')]


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def js(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,rows):path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
def arr(item,minimum=0):return {'type':'array','items':item,'minItems':minimum}
def string():return {'type':'string','minLength':1}
def enum(values):return {'type':'string','enum':list(values)}


def freeze_units(annotation,text):
    """Freeze the initial annotation's spans as operational units, not a gold parse."""
    units=[{'unit_id':'paragraph','kind':'paragraph','start':0,'end':len(text),'text':text,
            'sentence_ids':[],'segmentation_status':'operational_snapshot_not_verified_parse'}]
    sentences=[];cursor=0;span_number=0
    for index,sentence in enumerate(annotation['rhetorical_taxonomy']['sentence_structure'],1):
        quote=sentence['sentence_text'];start=text.find(quote,cursor)
        if start<0:raise ValueError('Cannot freeze exact ordered sentence')
        end=start+len(quote);cursor=end;sid=f'sentence_{index:02}'
        unit={'unit_id':sid,'kind':'sentence','start':start,'end':end,'text':quote,'sentence_ids':[sid],
              'segmentation_status':'operational_snapshot_not_verified_parse'}
        sentences.append(unit);units.append(unit)
        for clause in sentence['clauses']:
            fragment=clause['clause_text']
            if not fragment or quote.count(fragment)!=1:raise ValueError('Clause span is absent or ambiguous in its sentence')
            at=start+quote.find(fragment);span_number+=1
            units.append({'unit_id':f'span_{span_number:02}','kind':'subsentence','start':at,'end':at+len(fragment),
                'text':fragment,'sentence_ids':[sid],'segmentation_status':'operational_snapshot_not_verified_parse'})
    if ' '.join(' '.join(s['text'].split()) for s in sentences)!=' '.join(text.split()):raise ValueError('Frozen sentences do not cover text')
    for index,(a,b) in enumerate(zip(sentences,sentences[1:]),1):
        units.append({'unit_id':f'pair_{index:02}','kind':'sentence_pair','start':a['start'],'end':b['end'],
            'text':text[a['start']:b['end']],'sentence_ids':[a['unit_id'],b['unit_id']],
            'segmentation_status':'adjacency_only_no_assumed_rhetorical_relation'})
    if any(text[u['start']:u['end']]!=u['text'] for u in units):raise ValueError('Frozen unit/source mismatch')
    return units


def expected_cells(paragraph_id,units):
    return [{'cell_id':'cell_'+sha(paragraph_id+'|'+unit['unit_id']+'|'+fid)[:16],
             'unit_id':unit['unit_id'],'feature_id':fid,'canonical_id':canonical}
            for unit in units for fid,canonical,kind in FEATURES if unit['kind']==kind]


def output_schema(taxonomy_version,taxonomy_sha,protocol_sha):
    evidence=obj({'quote':string(),'occurrence':{'type':'integer','minimum':1}})
    observation=obj({'cell_id':string(),'unit_id':string(),'feature_id':enum([r[0] for r in FEATURES]),
        'canonical_id':enum([r[1] for r in FEATURES]),'state':enum(['present','absent','unresolved','not_applicable']),
        'evidence':arr(evidence),'claim_refs':arr(string()),'semantic_relation_refs':arr(string()),
        'binding_status':enum(['supported_by_supplied_context','uncertain','not_applicable']),
        'rationale':string(),'unit_or_binding_issues':arr(string())})
    observation['allOf']=[{'if':{'properties':{'state':{'const':'present'}}},'then':{'properties':{'evidence':{'minItems':1}}}}]
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:fama-style:fixed-unit-check:v1',
        **obj({'protocol_version':{'const':'fixed_unit_check_v1'},'protocol_sha256':{'const':protocol_sha},
            'taxonomy_version':{'const':taxonomy_version},'taxonomy_sha256':{'const':taxonomy_sha},
            'paragraph_id':string(),'unit_geometry_sha256':string(),
            'unit_assessments':arr(obj({'unit_id':string(),'suitability':enum(['usable','questionable','unusable']),'reason':string()}),1),
            'observations':arr(observation,1),
            'coverage_notes':arr(obj({'unit_id':string(),'description':string(),'evidence':arr(evidence)})),
            'semantic_verification':{'const':'not_performed'},'model_version':{'type':['string','null']},
            'needs_review':{'type':'boolean'},'limitations':arr(string())})}


def quote_span(text,quote,occurrence):
    if not quote or occurrence<1:raise ValueError('Invalid quote anchor')
    offset=0
    for _ in range(occurrence):
        start=text.find(quote,offset)
        if start<0:raise ValueError('Quote not found')
        offset=start+len(quote)
    return start,offset


def check_fixed_response(row,paragraph):
    """Mechanical cell completeness and source binding, in addition to JSON Schema."""
    errors=[];units={u['unit_id']:u for u in paragraph['units']};expected={c['cell_id']:c for c in paragraph['expected_cells']}
    for field in ('paragraph_id','unit_geometry_sha256'):
        if row[field]!=paragraph[field]:errors.append({'code':'wrong_paragraph_or_geometry','field':field})
    ids=[cell['cell_id'] for cell in row['observations']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):errors.append({'code':'missing_extra_or_duplicate_cells'})
    assessed=[unit['unit_id'] for unit in row['unit_assessments']]
    if len(assessed)!=len(set(assessed)) or set(assessed)!=set(units):errors.append({'code':'unit_assessment_coverage'})
    claims={c['claim_id'] for c in paragraph['semantic_context']['claims']}
    relations={r['relation_id'] for r in paragraph['semantic_context']['relations']}
    for cell in row['observations']:
        ident=cell['cell_id'];wanted=expected.get(ident)
        if wanted is None:continue
        if any(cell[key]!=wanted[key] for key in ('unit_id','feature_id','canonical_id')):
            errors.append({'code':'wrong_cell_feature_or_unit','cell_id':ident})
        if not set(cell['claim_refs'])<=claims:errors.append({'code':'unknown_claim_ref','cell_id':ident})
        if not set(cell['semantic_relation_refs'])<=relations:errors.append({'code':'unknown_relation_ref','cell_id':ident})
        unit=units[wanted['unit_id']]
        for evidence in cell['evidence']:
            try:start,end=quote_span(paragraph['text'],evidence['quote'],evidence['occurrence'])
            except ValueError:errors.append({'code':'missing_quote','cell_id':ident});continue
            if not unit['start']<=start<end<=unit['end']:errors.append({'code':'evidence_outside_fixed_unit','cell_id':ident})
    for note in row['coverage_notes']:
        if note['unit_id'] not in units:errors.append({'code':'unknown_coverage_unit'});continue
        unit=units[note['unit_id']]
        for evidence in note['evidence']:
            try:start,end=quote_span(paragraph['text'],evidence['quote'],evidence['occurrence'])
            except ValueError:errors.append({'code':'missing_coverage_quote'});continue
            if not unit['start']<=start<end<=unit['end']:errors.append({'code':'coverage_evidence_outside_fixed_unit'})
    return errors


def compare_fixed_responses(left,right):
    """Compare matching cells; uncertain/inapplicable agreement is not informative."""
    for field in ('unit_geometry_sha256','protocol_sha256','taxonomy_sha256'):
        if left[field]!=right[field]:raise ValueError('Cannot compare different units/protocols/taxonomies')
    def cells(row):return {(r['unit_id'],r['feature_id']):r for r in row['observations']}
    a,b=cells(left),cells(right)
    if set(a)!=set(b) or len(a)!=len(left['observations']) or len(b)!=len(right['observations']):
        raise ValueError('Incomplete or duplicated cell matrices')
    quality_a={u['unit_id']:u['suitability'] for u in left['unit_assessments']}
    quality_b={u['unit_id']:u['suitability'] for u in right['unit_assessments']}
    informative=[];usable=[];differences=[];table={}
    for key in sorted(a):
        if a[key]['canonical_id']!=b[key]['canonical_id']:raise ValueError('Canonical feature changed')
        x,y=a[key]['state'],b[key]['state'];table[x+' -> '+y]=table.get(x+' -> '+y,0)+1
        if x!=y:differences.append({'unit_id':key[0],'feature_id':key[1],'left':x,'right':y})
        if x in ('present','absent') and y in ('present','absent'):
            informative.append(x==y)
            if quality_a.get(key[0])==quality_b.get(key[0])=='usable':usable.append(x==y)
    return {'cells_compared':len(a),'state_matches':len(a)-len(differences),'state_transition_counts':table,
        'both_informative_cells':len(informative),'both_informative_matches':sum(informative),
        'informative_agreement':sum(informative)/len(informative) if informative else None,
        'both_usable_and_informative_cells':len(usable),'both_usable_and_informative_matches':sum(usable),
        'unit_quality_disagreements':sum(quality_a.get(k)!=quality_b.get(k) for k in set(quality_a)|set(quality_b)),
        'state_differences':differences,'semantic_accuracy':None,'training_readiness_established':False,
        'caveat':'Descriptive two-text consistency check. All-unresolved agreement is not informative; this metric is not comparable numerically to prior category-set Jaccard.'}


def build(output):
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new fixed-unit packet directory')
    base=ROOT/'research_runs/taxonomy_discovery_v1';assets=ROOT/'docs/fixed_unit_check_v1'
    key_path=base/'followup_v1/PRIVATE_SELECTION_AND_LINKS.json'
    taxonomy_path=base/'responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json'
    a_path=base/'responses/task_5_fresh_validation/canonical_annotations_A.jsonl';b_path=a_path.with_name('canonical_annotations_B.jsonl')
    input_paths=[key_path,taxonomy_path,a_path,b_path,assets/'ANNOTATION_PROMPT.md',assets/'START_HERE.md']
    for group in 'AB':input_paths.append(base/f'stage1_audit_v1/stage2_ready/batch_{group}/paragraphs.jsonl')
    before={str(p.relative_to(ROOT)):digest(p) for p in input_paths}
    audited=json.loads((base/'stage2_audit_v1/manifest.json').read_text())
    for path in (a_path,b_path):
        if digest(path)!=audited['input_response_hashes'][str(path.relative_to(ROOT))]:raise ValueError('Audited response changed')
    follow=json.loads((base/'followup_v1/manifest.json').read_text())
    if digest(key_path)!=follow['files'][key_path.name]:raise ValueError('Repeat key changed')
    references=json.loads(key_path.read_text())['fresh_annotations'];selected=[r['repeat_of'] for r in references if r['repeat_of']]
    if len(selected)!=2 or len(set(selected))!=2:raise ValueError('Expected the two original repeated texts')
    annotations={r['paragraph_id']:r for path in (a_path,b_path) for r in load(path)}
    texts={r['paragraph_id']:r['text'] for group in 'AB' for r in load(base/f'stage1_audit_v1/stage2_ready/batch_{group}/paragraphs.jsonl')}
    taxonomy=json.loads(taxonomy_path.read_text());taxonomy_hash=digest(taxonomy_path)
    expected_hashes={annotations[ident]['taxonomy_sha256'] for ident in selected}
    if expected_hashes!={taxonomy_hash}:raise ValueError('Taxonomy no longer matches the annotations')
    categories={row['canonical_id']:row for row in taxonomy['canonical_items']}
    features=[]
    for fid,cid,kind in FEATURES:
        category={k:copy.deepcopy(v) for k,v in categories[cid].items() if k not in ('supporting_inventory_ids','provisional_reason')}
        features.append({'feature_id':fid,'canonical_id':cid,'unit_kind':kind,'category':category,
            'measurement_scope':'Assess within the fixed unit, using full paragraph context; a positive is existence of supported instances, not permission to transform text.'})
    protocol={'protocol_version':'fixed_unit_check_v1','taxonomy_version':taxonomy['taxonomy_version'],'taxonomy_sha256':taxonomy_hash,
        'features':features,'unit_source':'Initial Stage 2 A/B operational spans, not selected or reconciled using repeat C labels',
        'interpretation':'Presence/absence check, not exhaustive taxonomy, gold segmentation or semantic verification',
        'applicability':'All expected cells must be returned. Use unresolved for ambiguity and not_applicable only when the unit is genuinely inapplicable.'}
    protocol_hash=sha(json.dumps(protocol,sort_keys=True,ensure_ascii=False));protocol['protocol_sha256']=protocol_hash
    schema=output_schema(taxonomy['taxonomy_version'],taxonomy_hash,protocol_hash)
    frozen={};linkage=[];counts=[]
    for source_id in selected:
        text=texts[source_id];annotation=annotations[source_id];units=freeze_units(annotation,text)
        geometry_hash=sha(json.dumps({'text_sha256':sha(text),'units':units},sort_keys=True,ensure_ascii=False))
        frozen[source_id]={'text':text,'units':units,'unit_geometry_sha256':geometry_hash,
            'semantic_context':copy.deepcopy(annotation['semantic_content']),
            'semantic_context_status':'Unverified prior interpretation; flag issues without editing claims/relations',
            'prior_uncertainty_notes':annotation['review_notes']}
    output.mkdir(parents=True,exist_ok=True)
    for round_number in (1,2):
        directory=output/f'round_{round_number}';directory.mkdir();paragraphs=[]
        for source_id in selected:
            ident='fu_'+sha(f'fixed_v1|{round_number}|{source_id}')[:16]
            paragraph={'paragraph_id':ident,**copy.deepcopy(frozen[source_id])}
            paragraph['expected_cells']=expected_cells(ident,paragraph['units']);paragraphs.append(paragraph)
            linkage.append({'round':round_number,'paragraph_id':ident,'source_id':source_id,
                'text_sha256':sha(paragraph['text']),'unit_geometry_sha256':paragraph['unit_geometry_sha256'],
                'cell_map':{row['cell_id']:{'unit_id':row['unit_id'],'feature_id':row['feature_id']} for row in paragraph['expected_cells']}})
            counts.append({'round':round_number,'paragraph_id':ident,'units':len(paragraph['units']),'cells':len(paragraph['expected_cells'])})
        random.Random(20261024+round_number).shuffle(paragraphs)
        lines(directory/'paragraphs.jsonl',paragraphs);js(directory/'feature_checklist.json',protocol);js(directory/'output.schema.json',schema)
        (directory/'ANNOTATION_PROMPT.md').write_bytes((assets/'ANNOTATION_PROMPT.md').read_bytes())
    (output/'START_HERE.md').write_bytes((assets/'START_HERE.md').read_bytes())
    js(output/'PRIVATE_LINKS_DO_NOT_UPLOAD.json',linkage)
    js(output/'preparation_summary.json',{'unique_texts':2,'independent_rounds_requested':2,'requested_annotation_rows':len(counts),
        'feature_types':len(FEATURES),'round_counts':counts,'cells_per_round':sum(r['cells'] for r in counts if r['round']==1),
        'source_unit_selection':'initial A/B snapshot; not a gold parse','taxonomy_modified':False,
        'new_model_calls':0,'gpu_jobs_submitted':0,'held_out_data_used':False})
    for name,h in before.items():
        if digest(ROOT/name)!=h:raise ValueError('Source changed')
    js(output/'manifest.json',{'input_hashes':before,'inputs_unchanged':True,'builder_sha256':digest(Path(__file__).resolve()),
        'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    (base/'responses/task_6_fixed_unit_check').mkdir(exist_ok=True)
    print(json.dumps({'unique_texts':2,'rounds':2,'features':len(FEATURES),'cells_per_round':sum(r['cells'] for r in counts if r['round']==1),'model_calls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir)
