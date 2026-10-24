"""Audit returned discovery files and prepare a private compact consolidation pack.

Run with Python and jsonschema installed. No models, credentials or held-out data.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_lines(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def normalized(text):return ' '.join(text.split())


def check_annotation(annotation,text):
    """Mechanical consistency only; matching evidence is not semantic verification."""
    errors=[];counts=collections.Counter();sem=annotation['semantic_content'];tax=annotation['rhetorical_taxonomy']
    sentences=tax['sentence_structure']
    namespaces={'entity_refs':[r['entity_id'] for r in sem['entities']],
        'claim_refs':[r['claim_id'] for r in sem['claims']],
        'semantic_relation_refs':[r['relation_id'] for r in sem['relations']],
        'from_move_refs':[r['move_id'] for r in tax['paragraph_structure']['moves']],
        'depends_on_clause_ids':[c['clause_id'] for s in sentences for c in s['clauses']],
        'sentence_ids':[s['sentence_id'] for s in sentences]}
    for key,ids in namespaces.items():
        if len(ids)!=len(set(ids)):errors.append({'code':'duplicate_ids','path':key})
    aliases={'scope_claim_refs':'claim_refs','source_claim_refs':'claim_refs','target_claim_refs':'claim_refs',
             'main_claim_refs':'claim_refs','to_move_refs':'from_move_refs'}
    def visit(value,path=''):
        if isinstance(value,dict):
            if 'quote' in value and 'occurrence' in value:
                counts['evidence_anchors']+=1
                if not value['quote'] or text.count(value['quote'])<value['occurrence']:
                    errors.append({'code':'evidence_not_found','path':path})
            for key,child in value.items():
                ns=aliases.get(key,key)
                if ns in namespaces and isinstance(child,list) and any(ref not in namespaces[ns] for ref in child):
                    errors.append({'code':'unresolved_reference','path':path+'/'+key})
                visit(child,path+'/'+key)
        elif isinstance(value,list):
            for i,child in enumerate(value):visit(child,path+'/'+str(i))
    visit(annotation)
    if normalized(' '.join(s['sentence_text'] for s in sentences))!=normalized(text):
        errors.append({'code':'sentence_coverage_mismatch','path':'/rhetorical_taxonomy/sentence_structure'})
    used=set()
    for i,sentence in enumerate(sentences):
        used.update(sentence['claim_refs'])
        if sentence['sentence_text'] not in text:errors.append({'code':'sentence_not_exact','path':str(i)})
        for clause in sentence['clauses']:
            if clause['clause_text'] not in sentence['sentence_text']:
                errors.append({'code':'clause_not_exact','path':str(i)+'/'+clause['clause_id']})
    if used!=set(namespaces['claim_refs']):errors.append({'code':'claims_not_bound_to_sentences','path':'sentence_structure'})
    counts.update(claims=len(sem['claims']),semantic_relations=len(sem['relations']),sentences=len(sentences))
    return errors,dict(counts)


def labels(annotation):
    """Extract every style label; exclude semantic predicates and qualification kinds."""
    tax=annotation['rhetorical_taxonomy'];base='/rhetorical_taxonomy/'
    paragraph=tax['paragraph_structure']
    for key,dimension in (('moves','move_role'),('progression','rhetorical_relation'),('qualification_placement','qualification_placement')):
        label_key={'moves':'role','progression':'relation','qualification_placement':'placement'}[key]
        for i,row in enumerate(paragraph[key]):
            yield 'paragraph',dimension,row[label_key],base+'paragraph_structure/'+key+'/'+str(i)+'/'+label_key,row
    for i,sentence in enumerate(tax['sentence_structure']):
        for j,row in enumerate(sentence['clauses']):
            owner={**row,'evidence':[{'quote':row['clause_text'],'occurrence':None,'anchor_kind':'clause_text_without_supplied_occurrence'}]}
            yield 'sentence','clause_role',row['role'],base+f'sentence_structure/{i}/clauses/{j}/role',owner
        for j,row in enumerate(sentence['features']):
            yield 'sentence',row['dimension'],row['label'],base+f'sentence_structure/{i}/features/{j}/label',row
    for i,row in enumerate(tax['language_features']):
        yield 'language',row['dimension'],row['label'],base+f'language_features/{i}/label',row


def inventory(annotations):
    groups=collections.defaultdict(list)
    for annotation,context,pair_id in annotations:
        for level,dimension,label,path,owner in labels(annotation):
            key=(level,dimension,label['name'],label['definition'])
            groups[key].append({'annotation_id':annotation['paragraph_id'],'json_path':path,'context':context,
                'pair_id':pair_id,'claim_refs':owner.get('claim_refs',[]),'evidence':owner.get('evidence',[]),
                'certainty':owner.get('certainty','interpretive'),'annotation_needs_review':annotation['needs_review']})
    result=[];occurrences=[]
    for index,(key,items) in enumerate(sorted(groups.items()),1):
        ident=f'L{index:04}';level,dimension,name,definition=key;examples=[]
        # Short evidence from both contexts where available, without truncating quotes.
        for context in sorted({item['context'] for item in items}):
            example=min((item for item in items if item['context']==context),
                        key=lambda row:(min((len(e['quote']) for e in row['evidence']),default=10**9),row['annotation_id'],row['json_path']))
            shortest=min(example['evidence'],key=lambda e:len(e['quote'])) if example['evidence'] else None
            examples.append({k:v for k,v in example.items() if k not in ('pair_id','evidence')})
            examples[-1]['evidence']=[] if shortest is None else [shortest]
        result.append({'inventory_id':ident,'level':level,'dimension':dimension,'provisional_label':name,
            'definition':definition,'occurrence_count':len(items),'distinct_annotation_count':len({r['annotation_id'] for r in items}),
            'distinct_pair_count':len({r['pair_id'] for r in items}),'context_counts':dict(collections.Counter(r['context'] for r in items)),
            'unresolved_occurrences':sum(r['certainty']=='unresolved' for r in items),
            'examples':examples,'evidence_note':'Representative exact quotes, not exhaustive evidence or independent semantic verification'})
        occurrences.extend({'inventory_id':ident,**row} for row in items)
    return result,occurrences


def audit(package,responses,output):
    import jsonschema
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new audit directory')
    source_files=[p for p in responses.rglob('*') if p.is_file()]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    inputs={r['paragraph_id']:r['text'] for r in load(package/'paragraphs_A.jsonl')}
    pairs={r['pair_id']:r for r in load(package/'neutral_reference_pairs.jsonl')}
    key= json.loads((package/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json').read_text())
    pair_by_original={r['original_item_id']:r['pair_id'] for r in key}
    schema_a=json.loads((package/'annotation_output.schema.json').read_text())
    schema_n=json.loads((package/'neutral_review_output.schema.json').read_text())
    a=[];n=[];findings=[];annotations=[];counts=collections.Counter()
    for kind,pattern,schema,destination in [('original','task_1_original_annotations/annotations_A_*.jsonl',schema_a,a),
                                             ('neutral','task_2_original_annotations/neutral_proposals_*.jsonl',schema_n,n)]:
        validator=jsonschema.Draft202012Validator(schema)
        for path in sorted(responses.glob(pattern)):
            rows=load(path);destination.extend(rows)
            # Preserve chunk identity, not just aggregate coverage.
            index=path.stem.rsplit('_',1)[-1]
            expected=load(package/(('paragraphs_A_' if kind=='original' else 'neutral_reference_pairs_')+index+'.jsonl'))
            idkey='paragraph_id' if kind=='original' else 'pair_id'
            if {r[idkey] for r in rows}!={r[idkey] for r in expected}:
                findings.append({'code':'chunk_id_mismatch','file':path.name})
            for line,row in enumerate(rows,1):
                for error in validator.iter_errors(row):
                    findings.append({'code':'schema_error','file':path.name,'line':line,'path':list(error.absolute_path),'validator':error.validator})
    for rows,idkey,expected in ((a,'paragraph_id',inputs),(n,'pair_id',pairs)):
        ids=[row[idkey] for row in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):findings.append({'code':'duplicate_or_missing_ids','kind':idkey})
    combined=responses/'task_2_original_annotations/neutral_proposals.jsonl'
    combined_matches=None
    if combined.exists():
        full=load(combined);combined_matches=len(full)==len(n) and {r['pair_id']:r for r in full}=={r['pair_id']:r for r in n}
        if not combined_matches:findings.append({'code':'combined_neutral_disagrees_with_chunks'})
    if findings:raise ValueError('Input/schema/coverage validation failed; no consolidation files created: '+json.dumps(findings))
    for row in a:
        errors,measure=check_annotation(row,inputs[row['paragraph_id']]);counts.update(measure)
        findings.extend({'annotation_id':row['paragraph_id'],**error} for error in errors)
        if row['provenance']['annotation_context']!='single_paragraph':findings.append({'code':'wrong_original_context','annotation_id':row['paragraph_id']})
        annotations.append((row,'original_observed',pair_by_original[row['paragraph_id']]))
    for row in n:
        pair=pairs[row['pair_id']]
        for k in ('reference_id','proposal_id'):
            if pair[k]!=row[k]:findings.append({'code':'changed_pair_binding','pair_id':row['pair_id'],'field':k})
        for issue in row['issues']:
            for field,source in (('original_evidence','original_paragraph'),('reference_evidence','plain_english_reference')):
                for evidence in issue[field]:
                    counts['issue_evidence_anchors']+=1
                    if pair[source].count(evidence['quote'])<evidence['occurrence']:
                        findings.append({'code':'issue_evidence_not_found','pair_id':row['pair_id'],'field':field})
        if row['decision']=='retain' and row['proposed_neutral_text']!=pair['plain_english_reference']:
            findings.append({'code':'retain_text_changed','pair_id':row['pair_id']})
        annotation=row['proposed_neutral_annotation']
        if annotation:
            if annotation['paragraph_id']!=row['proposal_id'] or annotation['provenance']['annotation_context']!='paired_neutral_design':
                findings.append({'code':'wrong_proposal_annotation_binding','pair_id':row['pair_id']})
            errors,measure=check_annotation(annotation,row['proposed_neutral_text']);counts.update(measure)
            findings.extend({'annotation_id':annotation['paragraph_id'],**error} for error in errors)
            annotations.append((annotation,'neutral_proposed_paired',row['pair_id']))
    output.mkdir(parents=True,exist_ok=True)
    summary={'original_annotations':len(a),'neutral_proposals':len(n),'decision_counts':dict(collections.Counter(r['decision'] for r in n)),
        'combined_neutral_file_matches_chunks':combined_matches,'duplicate_combined_rows_counted':False,
        'mechanical_findings':findings,'mechanically_ready_for_vocabulary_consolidation':not findings,
        'counts':dict(counts),'annotation_review_flags':{context:sum(row['needs_review'] for row,c,_ in annotations if c==context) for context in ('original_observed','neutral_proposed_paired')},
        'semantic_faithfulness_verified':False,'neutral_repairs_applied_to_dataset':False,'held_out_data_used':False,
        'human_review_required':False,'new_model_calls':0,
        'caveat':'Schema/evidence/reference checks establish mechanical consistency, not claim completeness, correct interpretation or repair faithfulness.'}
    write_json(output/'audit_summary.json',summary)
    if not findings:
        catalog,occurrences=inventory(annotations)
        upload=output/'consolidation_upload';upload.mkdir()
        write_lines(upload/'label_inventory.jsonl',catalog)
        summary['label_inventory_items']=len(catalog);summary['label_occurrences']=len(occurrences)
        write_json(upload/'discovery_summary.json',{**summary,'context_note':'Both sets come from the same 28 training pairs; paired neutral proposals are not blind independent annotations.'})
        write_lines(output/'occurrences.private.jsonl',occurrences)
        write_lines(output/'original_annotations.checked.private.jsonl',a)
        write_lines(output/'neutral_proposals.checked.private.jsonl',n)
        prompt=ROOT/'docs/taxonomy_discovery_v1/CONSOLIDATION_INVENTORY_PROMPT.md'
        (upload/'CONSOLIDATION_PROMPT.md').write_bytes(prompt.read_bytes())
        write_json(output/'audit_summary.json',summary)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items())
    write_json(output/'manifest.json',{'response_hashes':before,'responses_unchanged':True,
        'package_manifest_sha256':hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest(),
        'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ('original_annotations','neutral_proposals','decision_counts','mechanically_ready_for_vocabulary_consolidation')}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=ROOT/'research_runs/taxonomy_discovery_v1/package_v1')
    parser.add_argument('--responses',type=Path,default=ROOT/'research_runs/taxonomy_discovery_v1/responses')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();audit(args.package,args.responses,args.output_dir)
