"""Audit canonical annotations and compare masked repeats without conflating spans."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from audit_taxonomy_discovery import check_annotation,labels

ROOT=Path(__file__).resolve().parents[2]


def load(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def quote_span(text,quote,occurrence):
    offset=0
    if not quote or occurrence<1:raise ValueError('Invalid evidence anchor')
    for _ in range(occurrence):
        start=text.find(quote,offset)
        if start<0:raise ValueError('Evidence anchor not found')
        offset=start+len(quote)
    return start,offset


def label_records(annotation,text):
    starts=[];cursor=0;units=annotation['rhetorical_taxonomy']['sentence_structure']
    for unit in units:
        start=text.find(unit['sentence_text'],cursor)
        if start<0:raise ValueError('Sentence order/exactness mismatch')
        starts.append(start);cursor=start+len(unit['sentence_text'])
    result=[]
    for level,dimension,label,path,owner in labels(annotation):
        if '/clauses/' in path:
            index=int(path.split('/sentence_structure/')[1].split('/')[0])
            sentence=units[index]['sentence_text'];clause=owner['clause_text']
            anchors=None
            if sentence.count(clause)==1:
                at=starts[index]+sentence.find(clause);anchors=[(at,at+len(clause))]
        else:anchors=sorted({quote_span(text,e['quote'],e['occurrence']) for e in owner['evidence']})
        result.append({'annotation_id':annotation['paragraph_id'],'level':level,'dimension':dimension,'json_path':path,
            'status':label['status'],'canonical_id':label['canonical_id'],
            'candidate_canonical_ids':label['candidate_canonical_ids'],'anchors':anchors,
            'claim_refs':owner.get('claim_refs',[]),'semantic_relation_refs':owner.get('semantic_relation_refs',[])})
    return result


def check_canonical(annotation,text,categories):
    findings,counts=check_annotation(annotation,text)
    records=label_records(annotation,text)
    for row in records:
        ids=([row['canonical_id']] if row['canonical_id'] else [])+row['candidate_canonical_ids']
        for ident in ids:
            if ident not in categories:findings.append({'code':'unknown_canonical_id','path':row['json_path']});continue
            category=categories[ident]
            if category['level']!=row['level'] or category['dimension']!=row['dimension']:
                findings.append({'code':'wrong_canonical_level_or_dimension','path':row['json_path']})
        if len(row['candidate_canonical_ids'])!=len(set(row['candidate_canonical_ids'])):
            findings.append({'code':'duplicate_candidates','path':row['json_path']})
    claims={r['claim_id'] for r in annotation['semantic_content']['claims']}
    gaps=annotation['coverage_gaps']
    for index,gap in enumerate(gaps):
        if not set(gap['claim_refs'])<=claims:findings.append({'code':'unknown_gap_claim_ref','path':f'/coverage_gaps/{index}'})
        if not any(row['status']=='not_covered' and row['level']==gap['level'] and row['dimension']==gap['dimension'] for row in records):
            findings.append({'code':'gap_without_corresponding_label','path':f'/coverage_gaps/{index}'})
    for row in records:
        if row['status']=='not_covered' and not any(gap['level']==row['level'] and gap['dimension']==row['dimension'] for gap in gaps):
            findings.append({'code':'not_covered_without_gap','path':row['json_path']})
    return findings,counts,records


def compare_repeat(original,repeated,text):
    a=label_records(original,text);b=label_records(repeated,text)
    def category_counts(rows,level=None):
        return collections.Counter((r['level'],r['dimension'],r['canonical_id']) for r in rows
                                   if r['status']=='assigned' and (level is None or r['level']==level))
    x,y=category_counts(a),category_counts(b)
    def jaccard(s,t):return len(set(s)&set(t))/max(1,len(set(s)|set(t)))
    def anchor_groups(rows):
        groups=collections.defaultdict(set)
        for r in rows:
            if r['anchors'] is None:continue
            key=(r['level'],r['dimension'],tuple(tuple(span) for span in r['anchors']))
            state=r['canonical_id'] if r['status']=='assigned' else r['status']+':'+','.join(sorted(r['candidate_canonical_ids']))
            groups[key].add(state)
        return groups
    u,v=anchor_groups(a),anchor_groups(b);shared=set(u)&set(v)
    conflicts=[{'level':key[0],'dimension':key[1],'anchors':list(key[2]),'base':sorted(u[key]),'repeat':sorted(v[key])}
               for key in sorted(shared) if u[key]!=v[key]]
    return {'base_id':original['paragraph_id'],'repeat_id':repeated['paragraph_id'],
        'category_set_jaccard':jaccard(x,y),
        'category_count_dice':2*sum((x&y).values())/max(1,sum(x.values())+sum(y.values())),
        'category_jaccard_by_level':{level:jaccard(category_counts(a,level),category_counts(b,level)) for level in ('paragraph','sentence','language')},
        'base_anchor_groups':len(u),'repeat_anchor_groups':len(v),'shared_anchor_groups':len(shared),
        'shared_anchor_groups_with_identical_categories':len(shared)-len(conflicts),'same_anchor_category_differences':conflicts,
        'base_only_anchor_groups':len(set(u)-set(v)),'repeat_only_anchor_groups':len(set(v)-set(u)),
        'unanchored_clause_labels':sum(r['anchors'] is None for r in a+b),
        'base_claim_units':len(original['semantic_content']['claims']),'repeat_claim_units':len(repeated['semantic_content']['claims']),
        'base_semantic_relations':len(original['semantic_content']['relations']),'repeat_semantic_relations':len(repeated['semantic_content']['relations']),
        'base_coverage_gaps':len(original['coverage_gaps']),'repeat_coverage_gaps':len(repeated['coverage_gaps']),
        'interpretation':'Exploratory two-text repeat check, not accuracy or general reliability. Exact-anchor agreement is conditional on shared selected spans.'}


def main(output):
    import jsonschema
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new fresh-validation audit directory')
    root=ROOT/'research_runs/taxonomy_discovery_v1';packet=root/'stage1_audit_v1/stage2_ready';responses=root/'responses/task_5_fresh_validation'
    response_paths=sorted(p for p in responses.iterdir() if p.is_file());before={str(p.relative_to(ROOT)):digest(p) for p in response_paths}
    prior_manifest=json.loads((root/'stage1_audit_v1/manifest.json').read_text())
    pairing_path=root/'followup_v1/PRIVATE_SELECTION_AND_LINKS.json'
    key=json.loads(pairing_path.read_text())['fresh_annotations'];key_by_id={r['paragraph_id']:r for r in key}
    prepared_manifest=json.loads((root/'followup_v1/manifest.json').read_text())
    if digest(pairing_path)!=prepared_manifest['files'][pairing_path.name]:raise ValueError('Private repeat key changed')
    findings=[];all_records=[];annotations={};texts={};batches={};input_paths=[pairing_path]
    for letter in 'ABC':
        batch=packet/('batch_'+letter)
        for filename in ('paragraphs.jsonl','taxonomy_categories.json','canonical_annotation_output.schema.json','VALIDATION_PROMPT.md'):
            path=batch/filename
            if digest(path)!=prior_manifest['files'][str(path.relative_to(root/'stage1_audit_v1'))]:raise ValueError('Stage 2 source packet changed')
            input_paths.append(path)
        inputs={r['paragraph_id']:r['text'] for r in load(batch/'paragraphs.jsonl')};texts.update(inputs)
        response=responses/f'canonical_annotations_{letter}.jsonl';rows=load(response)
        validator=jsonschema.Draft202012Validator(json.loads((batch/'canonical_annotation_output.schema.json').read_text()))
        categories={r['canonical_id']:r for r in json.loads((batch/'taxonomy_categories.json').read_text())['canonical_items']}
        ids=[r.get('paragraph_id') for r in rows]
        if set(ids)!=set(inputs) or len(ids)!=len(set(ids)):findings.append({'code':'batch_id_coverage_mismatch','batch':letter})
        records=[];count=collections.Counter()
        for row in rows:
            ident=row.get('paragraph_id');errors=list(validator.iter_errors(row))
            for error in errors:findings.append({'annotation_id':ident,'code':'schema_error','path':list(error.absolute_path),'validator':error.validator})
            if errors or ident not in inputs:continue
            errors,measure,extracted=check_canonical(row,inputs[ident],categories)
            findings.extend({'annotation_id':ident,**error} for error in errors)
            count.update(measure);annotations[ident]=row
            for item in extracted:item['batch']=letter;item['masked_repeat']=bool(key_by_id[ident]['repeat_of'])
            records.extend(extracted)
        statuses=dict(collections.Counter(r['status'] for r in records))
        batches[letter]={'rows':len(rows),'label_statuses':statuses,'counts':dict(count),
            'unique_assigned_categories':len({r['canonical_id'] for r in records if r['status']=='assigned'}),
            'coverage_gaps':sum(len(r['coverage_gaps']) for r in rows),'needs_review_rows':sum(r['needs_review'] for r in rows),
            'report_present':(responses/f'coverage_report_{letter}.md').exists()}
        all_records.extend(records)
    if set(annotations)!=set(key_by_id):findings.append({'code':'overall_annotation_coverage_mismatch'})
    repeats=[]
    if not findings:
        for row in key:
            if row['repeat_of']:
                if texts[row['paragraph_id']]!=texts[row['repeat_of']]:raise ValueError('Masked repeat text mismatch')
                repeats.append(compare_repeat(annotations[row['repeat_of']],annotations[row['paragraph_id']],texts[row['paragraph_id']]))
    summary={'annotation_rows':len(annotations),'unique_paragraphs':len({text for text in texts.values()}),'underlying_training_pairs':7,
        'masked_repeats':len(repeats),'batch_summaries':batches,'findings':findings,'mechanical_validation_passed':not findings,
        'fresh_primary_label_statuses':dict(collections.Counter(r['status'] for r in all_records if not r['masked_repeat'])),
        'fresh_primary_categories_observed':len({r['canonical_id'] for r in all_records if not r['masked_repeat'] and r['status']=='assigned'}),
        'repeat_comparisons':repeats,'coverage_gap_instances':[{'annotation_id':row['paragraph_id'],'masked_repeat':bool(key_by_id[row['paragraph_id']]['repeat_of']),
            'level':gap['level'],'dimension':gap['dimension'],'description':gap['description']} for row in annotations.values() for gap in row['coverage_gaps']],
        'taxonomy_frozen':False,'ready_for_mapping_training':False,'semantic_faithfulness_verified':False,'new_model_calls':0,'held_out_data_used':False,
        'decision':'Standardize annotation-unit selection and explicit present/absent/unresolved coverage before treating source-target structural differences as style.',
        'caveat':'High label coverage is not semantic accuracy. Two repeats cannot establish general reliability; agreement on shared anchors omits differently selected spans.'}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'audit_summary.json',summary)
    (output/'canonical_label_records.private.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in all_records))
    for name,h in before.items():
        if digest(ROOT/name)!=h:raise ValueError('Original response changed')
    write_json(output/'manifest.json',{'input_response_hashes':before,'responses_unchanged':True,
        'prepared_input_hashes':{str(path.relative_to(ROOT)):digest(path) for path in input_paths},'auditor_sha256':digest(Path(__file__).resolve()),
        'files':{str(path.relative_to(output)):digest(path) for path in output.rglob('*') if path.is_file()}})
    print(json.dumps({name:summary[name] for name in ('annotation_rows','unique_paragraphs','fresh_primary_label_statuses','mechanical_validation_passed')}))
    if findings:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path)
    main(parser.parse_args().output_dir)
