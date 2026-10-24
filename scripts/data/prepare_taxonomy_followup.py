"""Prepare context-resolution and fresh canonical-annotation packets; no inference."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from build_taxonomy_discovery import PAPERS,TRAIN_SHA256,tags,words

ROOT=Path(__file__).resolve().parents[2]
SEED=20261023


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def js(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(path,rows):path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
def arr(item,minimum=0):return {'type':'array','items':item,'minItems':minimum}
def text():return {'type':'string','minLength':1}
def enum(values):return {'type':'string','enum':list(values)}


def pointer(value,path):
    for part in path.lstrip('/').split('/'):
        part=part.replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


def case_id(row):return 'case_'+sha(row['inventory_id']+'|'+row['annotation_id']+'|'+row['json_path'])[:16]


def evidence_words(row,annotations):
    owner=pointer(annotations[row['annotation_id']],row['json_path'].rsplit('/',1)[0])
    snippet=' '.join(e['quote'] for e in owner.get('evidence',[])) or owner.get('clause_text','')
    return set(re.findall(r'\w+',snippet.casefold()))


def select_resolution_cases(occurrences,categories,annotations,paper_by_pair,limit=3):
    groups=collections.defaultdict(list)
    for row in occurrences:
        if row['mapping_status']=='split_requires_context' and any(categories[c]['control_status']=='candidate_presentation_control' for c in row['candidate_canonical_ids']):
            groups[row['inventory_id']].append(row)
    chosen_by_group={}
    for ident,group in sorted(groups.items()):
        chosen=[];remaining=sorted(group,key=case_id)
        median=statistics.median(len(evidence_words(row,annotations)) for row in group)
        while remaining and len(chosen)<limit:
            seen_papers={paper_by_pair[row['pair_id']] for row in chosen};seen_pairs={row['pair_id'] for row in chosen}
            seen_annotations={row['annotation_id'] for row in chosen}
            def rank(row):
                observed=evidence_words(row,annotations)
                distances=[1-len(observed&evidence_words(other,annotations))/max(1,len(observed|evidence_words(other,annotations))) for other in chosen]
                return (paper_by_pair[row['pair_id']] not in seen_papers,row['pair_id'] not in seen_pairs,
                        row['annotation_id'] not in seen_annotations,min(distances,default=0),
                        -abs(len(observed)-median),case_id(row))
            row=max(remaining,key=rank);chosen.append(row);remaining.remove(row)
        chosen_by_group[ident]=chosen
    # Round-robin groups, rather than a whole batch devoted to one broad label.
    selected=[group[round_index] for round_index in range(limit) for group in chosen_by_group.values() if len(group)>round_index]
    return selected,{key:len(value) for key,value in groups.items()}


def select_fresh(rows,excluded_ids,excluded_texts):
    if any(row.get('split')!='train' for row in rows):raise ValueError('Fresh taxonomy samples must be training-only')
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate training IDs')
    if set(row['paper_key'] for row in rows)!=set(PAPERS):raise ValueError('Unexpected training paper inventory')
    old=[row for row in rows if row['id'] in excluded_ids];selected=[]
    forbidden={' '.join(t.split()) for t in excluded_texts}
    for paper in PAPERS:
        previous=[row for row in old if row['paper_key']==paper]
        candidates=[row for row in rows if row['paper_key']==paper and row['id'] not in excluded_ids
                    and all(' '.join(row[field].split()) not in forbidden for field in ('completion','plain_english'))]
        if not previous or not candidates:raise ValueError('Need fresh and discovery examples in every training paper')
        def distance(a,b):
            ta,tb=tags(a),tags(b)
            length=abs(words(a['completion'])-words(b['completion']))/max(1,words(a['completion']),words(b['completion']))
            return length+len(ta^tb)/max(1,len(ta|tb))
        row=max(candidates,key=lambda candidate:(min(distance(candidate,prior) for prior in previous),sha(str(SEED)+candidate['id'])))
        selected.append(row)
    return selected


def resolution_schema(ids,version,digest):
    evidence=obj({'quote':text(),'occurrence':{'type':'integer','minimum':1}})
    binding=obj({'canonical_id':enum(ids),'evidence':arr(evidence,1),'claim_refs':arr(text()),'semantic_relation_refs':arr(text())})
    schema=obj({'resolution_version':{'const':'rhetorical_resolution_v1'},'taxonomy_version':{'const':version},
        'taxonomy_sha256':{'const':digest},'case_id':text(),'inventory_id':text(),'context_id':text(),
        'status':enum(['resolved_single','resolved_components','unresolved','taxonomy_gap']),
        'assignments':arr(binding),'remaining_candidate_ids':arr(enum(ids)),
        'rationale':text(),'gap_description':{'type':['string','null']},'annotation_issues':arr(text()),'needs_review':{'type':'boolean'}})
    schema.update({'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'urn:fama-style:rhetorical-resolution:v1',
        'allOf':[{'if':{'properties':{'status':{'const':'resolved_single'}}},'then':{'properties':{'assignments':{'minItems':1,'maxItems':1},'remaining_candidate_ids':{'maxItems':0}}}},
                 {'if':{'properties':{'status':{'const':'resolved_components'}}},'then':{'properties':{'assignments':{'minItems':2},'remaining_candidate_ids':{'maxItems':0}}}},
                 {'if':{'properties':{'status':{'enum':['unresolved','taxonomy_gap']}}},'then':{'properties':{'assignments':{'maxItems':0}}}},
                 {'if':{'properties':{'status':{'const':'taxonomy_gap'}}},'then':{'properties':{'gap_description':text()}},'else':{'properties':{'gap_description':{'type':'null'}}}}]})
    return schema


def canonical_schema(base,ids,version,digest):
    result=copy.deepcopy(base)
    label=obj({'canonical_id':{'anyOf':[enum(ids),{'type':'null'}]},
        'status':enum(['assigned','unresolved','not_covered']),'candidate_canonical_ids':arr(enum(ids)),'reason':text()})
    label['allOf']=[{'if':{'properties':{'status':{'const':'assigned'}}},'then':{'properties':{'canonical_id':enum(ids),'candidate_canonical_ids':{'maxItems':0}}},
                    'else':{'properties':{'canonical_id':{'type':'null'}}}},
                   {'if':{'properties':{'status':{'const':'not_covered'}}},'then':{'properties':{'candidate_canonical_ids':{'maxItems':0}}}}]
    result['$defs']['label']=label
    result['$id']='urn:fama-style:rhetorical-validation:annotation:v1'
    result['properties']['annotation_version']={'const':'rhetorical_validation_v1'}
    result['properties']['taxonomy_version']={'const':version};result['properties']['taxonomy_sha256']={'const':digest}
    result['properties']['provenance']['properties']['annotation_context']={'const':'single_paragraph'}
    result['properties']['coverage_gaps']=arr(obj({'level':enum(['paragraph','sentence','language']),
        'dimension':text(),'description':text(),'claim_refs':arr(text()),'evidence':arr({'$ref':'#/$defs/evidence'},1)}))
    result['required']+=['taxonomy_version','taxonomy_sha256','coverage_gaps']
    return result


def catalogue(items):
    # Keep definitions and safeguards verbatim, omit discovery examples/support metadata.
    return [{key:copy.deepcopy(value) for key,value in row.items() if key not in ('supporting_inventory_ids','provisional_reason')} for row in items]


def make_context(annotation,source,context):
    return {'context_id':'ctx_'+sha(annotation['paragraph_id']+'|'+sha(source))[:16],
        'annotation_id':annotation['paragraph_id'],'annotation_context':context,'text':source,'text_sha256':sha(source),
        'semantic_content':copy.deepcopy(annotation['semantic_content']),
        'sentence_units':[{'sentence_id':s['sentence_id'],'sentence_text':s['sentence_text'],'claim_refs':s['claim_refs'],
            'clauses':[{k:v for k,v in c.items() if k!='role'} for c in s['clauses']]} for s in annotation['rhetorical_taxonomy']['sentence_structure']],
        'paragraph_units':[{k:v for k,v in move.items() if k!='role'} for move in annotation['rhetorical_taxonomy']['paragraph_structure']['moves']],
        'annotation_needs_review':annotation['needs_review'],'annotation_review_notes':annotation['review_notes'],
        'semantic_verification_status':'unverified annotation evidence; original text remains authoritative for this task'}


def build(output):
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new follow-up directory; existing packets are immutable')
    base=ROOT/'research_runs/taxonomy_discovery_v1';package=base/'package_v1';source_a=base/'response_audit_v1';source_b=base/'response_audit_v2';audit=base/'consolidation_audit_v2'
    taxonomy_path=base/'responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json'
    train_path=ROOT/'data/fama_corpus_v1/train.jsonl';asset_dir=ROOT/'docs/taxonomy_validation_v1'
    sources=[taxonomy_path,train_path,package/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json',package/'paragraphs_A.jsonl',package/'paragraphs_B.jsonl',
        source_a/'original_annotations.checked.private.jsonl',source_a/'neutral_proposals.checked.private.jsonl',
        source_b/'annotations_B.normalized.private.jsonl',source_b/'consolidation_upload/label_inventory.jsonl',
        audit/'needs_context.private.jsonl',ROOT/'docs/taxonomy_discovery_v1/annotation_output.schema.json']
    sources+=list(asset_dir.glob('*.md'))
    before={str(path.relative_to(ROOT)):file_sha(path) for path in sources}
    if file_sha(train_path)!=TRAIN_SHA256:raise ValueError('Frozen train hash changed')
    for folder,names in ((package,['PRIVATE_MAPPING_DO_NOT_UPLOAD.json','paragraphs_A.jsonl','paragraphs_B.jsonl']),
        (source_a,['original_annotations.checked.private.jsonl','neutral_proposals.checked.private.jsonl']),
        (source_b,['annotations_B.normalized.private.jsonl','consolidation_upload/label_inventory.jsonl']),
        (audit,['needs_context.private.jsonl'])):
        saved=json.loads((folder/'manifest.json').read_text())['files']
        for name in names:
            if file_sha(folder/name)!=saved[name]:raise ValueError('Previously audited artifact changed')
    audited_hashes=json.loads((audit/'manifest.json').read_text())['input_hashes']
    hashes=[value for path,value in audited_hashes.items() if Path(path).name==taxonomy_path.name]
    if hashes!=[file_sha(taxonomy_path)]:raise ValueError('Taxonomy differs from the checked v2 proposal')
    taxonomy=json.loads(taxonomy_path.read_text());version=taxonomy['taxonomy_version'];digest=file_sha(taxonomy_path)
    categories={row['canonical_id']:row for row in taxonomy['canonical_items']};inventory={row['inventory_id']:row for row in read_lines(source_b/'consolidation_upload/label_inventory.jsonl')}
    key=json.loads((package/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json').read_text());paper_by_pair={row['pair_id']:row['paper_key'] for row in key}
    a_text={row['paragraph_id']:row['text'] for row in read_lines(package/'paragraphs_A.jsonl')}
    b_text={row['paragraph_id']:row['text'] for row in read_lines(package/'paragraphs_B.jsonl')}
    original=read_lines(source_a/'original_annotations.checked.private.jsonl');proposals=read_lines(source_a/'neutral_proposals.checked.private.jsonl');existing=read_lines(source_b/'annotations_B.normalized.private.jsonl')
    annotations={};contexts={}
    for annotation,source,context in ([(row,a_text[row['paragraph_id']],'original_observed') for row in original]+
        [(row['proposed_neutral_annotation'],row['proposed_neutral_text'],'neutral_proposed_paired') for row in proposals if row['proposed_neutral_annotation']]+
        [(row,b_text[row['paragraph_id']],'existing_reference_diagnostic') for row in existing]):
        annotations[annotation['paragraph_id']]=annotation;contexts[annotation['paragraph_id']]=make_context(annotation,source,context)
    unresolved=read_lines(audit/'needs_context.private.jsonl')
    if any(row['canonical_id'] is not None for row in unresolved):raise ValueError('Ambiguous occurrence was already assigned')
    selected,group_counts=select_resolution_cases(unresolved,categories,annotations,paper_by_pair)
    if len({case_id(row) for row in selected})!=len(selected):raise ValueError('Duplicate resolution case')
    output.mkdir(parents=True,exist_ok=True);case_records=[];selection_notes=[]
    for row in selected:
        inv=inventory[row['inventory_id']];ann=annotations[row['annotation_id']]
        actual_label=pointer(ann,row['json_path'])
        if actual_label!={'name':inv['provisional_label'],'definition':inv['definition']}:raise ValueError('Occurrence path/label mismatch')
        case_records.append({'case_id':case_id(row),'inventory_id':row['inventory_id'],'context_id':contexts[row['annotation_id']]['context_id'],
            'annotation_path':row['json_path'],'level':inv['level'],'dimension':inv['dimension'],
            'provisional_label':actual_label,'target_node':copy.deepcopy(pointer(ann,row['json_path'].rsplit('/',1)[0])),
            'candidate_canonical_ids':row['candidate_canonical_ids'],'disambiguation_rule':row['disambiguation_rule']})
        selection_notes.append({'case_id':case_id(row),'inventory_id':row['inventory_id'],'annotation_id':row['annotation_id'],
            'pair_id':row['pair_id'],'paper_key':paper_by_pair[row['pair_id']],'context':row['context']})
    context_by_id={row['context_id']:row for row in contexts.values()}
    resolution_root=output/'01_context_resolution';resolution_root.mkdir()
    for offset in range(0,len(case_records),7):
        batch=resolution_root/f'batch_{offset//7+1:02}';batch.mkdir();cases=case_records[offset:offset+7]
        ids=sorted({c for row in cases for c in row['candidate_canonical_ids']})
        lines(batch/'cases.jsonl',cases)
        lines(batch/'contexts.jsonl',[context_by_id[ident] for ident in sorted({row['context_id'] for row in cases})])
        js(batch/'candidate_categories.json',{'resolution_version':'rhetorical_resolution_v1','taxonomy_version':version,'taxonomy_sha256':digest,
            'canonical_items':catalogue([categories[c] for c in ids]),'note':'Derived definition-only excerpt; related_item_ids may point outside this excerpt. Use only the supplied candidates for each case.'})
        js(batch/'resolution_output.schema.json',resolution_schema(ids,version,digest))
        (batch/'RESOLUTION_PROMPT.md').write_bytes((asset_dir/'RESOLUTION_PROMPT.md').read_bytes())
    train=read_lines(train_path);excluded={row['training_id'] for row in key}
    fresh=select_fresh(train,excluded,[row['text'] for row in contexts.values()])
    fresh_root=output/'02_fresh_validation';fresh_root.mkdir();batches={'A':[],'B':[]};fresh_key=[]
    for row in fresh:
        for group,field in (('A','completion'),('B','plain_english')):
            ident='v_'+sha(f'{SEED}|{group}|{row["id"]}')[:16]
            batches[group].append({'paragraph_id':ident,'text':row[field]})
            fresh_key.append({'paragraph_id':ident,'training_id':row['id'],'paper_key':row['paper_key'],'group':group,'text_sha256':sha(row[field]),'repeat_of':None})
    # Two repeats from different papers, run in an independent conversation with new IDs.
    repeat_sources=[next(row for row in fresh_key if row['group']=='A' and row['paper_key']==PAPERS[0]),
                    next(row for row in fresh_key if row['group']=='B' and row['paper_key']==PAPERS[-1])]
    source_text={r['paragraph_id']:r['text'] for batch in batches.values() for r in batch};batches['C']=[]
    for source in repeat_sources:
        ident='v_'+sha(f'{SEED}|repeat|{source["paragraph_id"]}')[:16]
        batches['C'].append({'paragraph_id':ident,'text':source_text[source['paragraph_id']]})
        fresh_key.append({**source,'paragraph_id':ident,'group':'C','repeat_of':source['paragraph_id']})
    schema=canonical_schema(json.loads((ROOT/'docs/taxonomy_discovery_v1/annotation_output.schema.json').read_text()),sorted(categories),version,digest)
    for index,(name,rows) in enumerate(batches.items()):
        random.Random(SEED+index).shuffle(rows);batch=fresh_root/('batch_'+name);batch.mkdir()
        lines(batch/'paragraphs.jsonl',rows)
        js(batch/'taxonomy_categories.json',{'taxonomy_version':version,'taxonomy_sha256':digest,'canonical_items':catalogue(taxonomy['canonical_items']),
            'status':'proposed_not_frozen; evaluate coverage and ambiguity without changing definitions'})
        js(batch/'canonical_annotation_output.schema.json',schema)
        (batch/'VALIDATION_PROMPT.md').write_bytes((asset_dir/'VALIDATION_PROMPT.md').read_bytes())
    (output/'START_HERE.md').write_bytes((asset_dir/'START_HERE.md').read_bytes())
    js(output/'PRIVATE_SELECTION_AND_LINKS.json',{'resolution_cases':selection_notes,'fresh_annotations':fresh_key,
        'fresh_pair_training_ids':[row['id'] for row in fresh],'excluded_discovery_training_ids':sorted(excluded)})
    summary={'selected_resolution_cases':len(selected),'control_related_ambiguous_inventory_groups':len(group_counts),
        'control_related_occurrences_in_source':sum(group_counts.values()),'all_original_ambiguous_occurrences_still_unassigned':len(unresolved),
        'selected_per_inventory':dict(collections.Counter(row['inventory_id'] for row in selected)),
        'sampling':'Up to three cases per group; different papers/pairs/annotations, then evidence-wording diversity; not predicted category labels',
        'fresh_training_pairs':len(fresh),'fresh_unique_paragraphs':len(fresh_key)-len(repeat_sources),'masked_repeat_annotations':len(repeat_sources),
        'fresh_papers':dict(collections.Counter(row['paper_key'] for row in fresh)),
        'fresh_to':'taxonomy discovery only; these remain members of the existing training split',
        'taxonomy_version':version,'taxonomy_sha256':digest,'taxonomy_modified':False,'source_annotations_modified':False,
        'label_decisions_made':0,'semantic_repairs_applied':0,'validation_or_test_rows_read':0,'new_model_calls':0,'gpu_jobs_submitted':0}
    js(output/'preparation_summary.json',summary)
    for name,digest_before in before.items():
        if file_sha(ROOT/name)!=digest_before:raise ValueError('Source changed during packet preparation')
    js(output/'manifest.json',{'implementation':'taxonomy_followup_v1','input_hashes':before,'inputs_unchanged':True,
        'code_hashes':{str(path.relative_to(ROOT)):file_sha(path) for path in [Path(__file__).resolve(),ROOT/'scripts/data/build_taxonomy_discovery.py']},
        'files':{str(path.relative_to(output)):file_sha(path) for path in sorted(output.rglob('*')) if path.is_file()}})
    for name in ('task_4_context_resolution','task_5_fresh_validation'):(base/'responses'/name).mkdir(exist_ok=True)
    print(json.dumps({key:summary[key] for key in ('selected_resolution_cases','fresh_training_pairs','fresh_unique_paragraphs','masked_repeat_annotations','new_model_calls')}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    build(parser.parse_args().output_dir)
