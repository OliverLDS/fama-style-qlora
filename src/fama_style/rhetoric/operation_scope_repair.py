"""Recover declared local programs as provenance-bound contiguous excerpt pairs.

Preserves raw annotations and whole paragraphs. Does not relax the original
whole-paragraph coverage validator or claim broader operation-family support.
"""
import argparse
import copy
import json
from pathlib import Path
import re
from . import operation_expansion as base
from .experiment import fresh,write_json,write_lines
from .concise_plan import file_sha,read_lines
from fama_style.research.claims import sha

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='operation_scope_recovery_v1'
SPECS=[{'case_id':'Xaf34a19c34be2743','source_units':['S2','S3'],'target_units':['T2','T3','T4','T5'],'claim_ids':['c2','c3','c4'],'drop_negation_claim':None},
 {'case_id':'X97176a208d46cd48','source_units':['S1','S2'],'target_units':['T1','T2','T3'],'claim_ids':['c1','c2'],'drop_negation_claim':'c2'}]


def excerpt(record,side,unit_ids):
    units=record[side+'_units'];indices=[i for i,u in enumerate(units) if u['unit_id'] in unit_ids]
    if not indices or [units[i]['unit_id'] for i in indices]!=unit_ids or indices!=list(range(min(indices),max(indices)+1)):raise ValueError('Excerpt units must be contiguous and ordered')
    text=record['source' if side=='source' else 'reference'];positions=[];cursor=0
    for unit in units:
        start=text.find(unit['text'],cursor)
        if start<0:raise ValueError('Unit not found in original text')
        positions.append((start,start+len(unit['text'])));cursor=positions[-1][1]
    start=positions[indices[0]][0];end=positions[indices[-1]][1]
    return text[start:end],{'start':start,'end':end,'unit_ids':unit_ids},[units[i] for i in indices]


def rebase(items,parent_text,start,end):
    text=parent_text[start:end];result=[]
    for item in items:
        bounds=base.grounded.span(parent_text,item['quote'],item['occurrence'])
        if not start<=bounds['start']<bounds['end']<=end:raise ValueError('Evidence outside excerpt; no silent clipping')
        local_start=bounds['start']-start
        matches=list(re.finditer(re.escape(item['quote']),text))
        occurrence=next((i for i,m in enumerate(matches,1) if m.start()==local_start),None)
        if occurrence is None:raise ValueError('Cannot rebase exact evidence')
        result.append({'quote':item['quote'],'occurrence':occurrence})
    return result


def recover(record,annotation,spec):
    if record['case_id']!=annotation['case_id'] or spec['case_id']!=annotation['case_id']:raise ValueError('Case mismatch')
    if annotation['pair_sha256']!=record['pair_sha256']:raise ValueError('Changed original pair')
    selected=set(spec['claim_ids']);derived=copy.deepcopy(record);a=copy.deepcopy(annotation);bounds={};changes=[]
    for side in ('source','target'):
        text,bounds[side],units=excerpt(record,side,spec[side+'_units'])
        derived[side+'_units']=units;derived['source' if side=='source' else 'reference']=text
        picked=[u for u in a[side+'_units'] if u['unit_id'] in spec[side+'_units']]
        if len(picked)!=len(units) or any(not set(u['claim_refs'])<=selected for u in picked):raise ValueError('Selected unit contains unretained claims')
        a[side+'_units']=picked
    derived['case_id']='L'+sha(PROTOCOL+'|'+record['case_id'])[:16]
    derived['pair_sha256']=base.pair_sha({'plain_english':derived['source'],'completion':derived['reference']})
    a['case_id']=derived['case_id'];a['pair_sha256']=derived['pair_sha256'];a['claims']=[c for c in a['claims'] if c['claim_id'] in selected]
    if {c['claim_id'] for c in a['claims']}!=selected:raise ValueError('Unknown selected claim')
    def bound(items,side):
        parent=record['source' if side=='source' else 'reference'];b=bounds[side]
        return rebase(items,parent,b['start'],b['end'])
    for claim in a['claims']:
        for side in ('source','target'):
            claim[side+'_evidence']=bound(claim[side+'_evidence'],side)
            for protected in claim['protected']:protected[side+'_evidence']=bound(protected[side+'_evidence'],side)
        if claim['claim_id']==spec.get('drop_negation_claim'):
            removed=[p for p in claim['protected'] if p['kind']=='negation']
            if len(removed)!=1:raise ValueError('Expected one misclassified negation item')
            spans=' '.join(e['quote'] for p in removed for s in ('source','target') for e in p[s+'_evidence'])
            if re.search(r'\b(not|no|never|neither|nor|without|cannot)\b',spans,re.I):raise ValueError('Negation item requires reassessment')
            claim['protected']=[p for p in claim['protected'] if p['kind']!='negation']
            changes.append({'field':'claims/'+claim['claim_id']+'/protected','action':'remove_one_misclassified_negation_tag',
                'reason':'The anchored evidence states a positive-mean inference and uncertainty above/below an estimate. Zero/above/below do not negate the proposition. Quantity, comparison and uncertainty tags remain.','removed':removed})
    kept=[];dropped=[]
    for relation in a['relations']:
        if set(relation['from_claim_refs']+relation['to_claim_refs'])<=selected:
            relation['source_evidence']=bound(relation['source_evidence'],'source');kept.append(relation)
        else:dropped.append(relation['relation_id'])
    a['relations']=kept
    a['program']['preserve_relation_refs']=[r for r in a['program']['preserve_relation_refs'] if r in {v['relation_id'] for v in kept}]
    for stage in a['program']['stages']:
        if not set(stage['claim_refs'])<=selected:raise ValueError('Program extends outside declared excerpt')
        stage['target_evidence']=bound(stage['target_evidence'],'target')
    a['notes']+=' Derived contiguous excerpt only; excluded parent claims remain archived, not part of this rewriting task. Original annotation is unchanged.'
    compiled=base.validate_annotation(derived,a)
    if not compiled['packing_or_order_contrast']:raise ValueError('No recovered contrast')
    return {'record':derived,'annotation':a,'compiled':compiled,'provenance':{'protocol':PROTOCOL,'parent_case_id':record['case_id'],
        'parent_pair_sha256':record['pair_sha256'],'parent_annotation_sha256':sha(json.dumps(annotation,sort_keys=True,ensure_ascii=False)),
        'source_scope':bounds['source'],'target_scope':bounds['target'],'selected_claim_ids':spec['claim_ids'],
        'excluded_claim_ids':sorted({c['claim_id'] for c in annotation['claims']}-selected),
        'excluded_relation_ids':dropped,'local_annotation_repairs':changes,'parent_paragraphs_and_raw_annotations_changed':False,
        'excerpt_text_rewritten':False,'semantic_equivalence_verified':False}}


def run(package,responses,original_audit,output):
    key=base.check_package(package);infos={r['case_id']:r for r in key['records']}
    for name,h in json.loads((original_audit/'manifest.json').read_text())['files'].items():
        if file_sha(original_audit/name)!=h:raise ValueError('Original audit artifact changed')
    coverage=json.loads((original_audit/'coverage.json').read_text())
    for name,h in coverage['response_hashes'].items():
        if file_sha(responses/name)!=h or file_sha(original_audit/'raw'/name)!=h:raise ValueError('Raw response changed')
    raw={}
    for name in coverage['response_hashes']:
        for a in read_lines(responses/name):
            if a['case_id'] in raw:raise ValueError('Duplicate case')
            raw[a['case_id']]=a
    results=[]
    for spec in SPECS:
        a=raw[spec['case_id']];info=infos[spec['case_id']]
        try:base.validate_annotation(info['record'],a)
        except ValueError as exc:
            if str(exc)!='program_omits_claims':raise ValueError('Unexpected original defect') from None
        else:raise ValueError('Original no longer has the documented coverage defect')
        item=recover(info['record'],a,spec);item['source_id']=info['source_id'];item['paper_key']=info['paper_key'];results.append(item)
    primary=read_lines(original_audit/'primary_pairs.provisional.private.jsonl');variants=read_lines(original_audit/'related_variants.private.jsonl')
    original_paragraph_gate=base.pilot_gates(primary,variants)
    for item in results:
        a=item['annotation'];r=item['record'];pair={'case_id':r['case_id'],'source_id':item['source_id'],'paper_key':item['paper_key'],'family':a['family'],'variant':a['variant'],
            'source':r['source'],'reference':r['reference'],'prompts':base.render_pair(r,item['compiled']),'unit':'contiguous_excerpt',
            'provenance':item['provenance'],'semantic_equivalence_verified':False}
        (primary if a['variant']=='seed_exact' else variants).append(pair)
    fresh(output);write_lines(output/'scoped_cards.private.jsonl',results);write_lines(output/'primary_pairs.provisional.private.jsonl',primary);write_lines(output/'related_variants.private.jsonl',variants)
    write_json(output/'original_paragraph_pilot_gate.json',original_paragraph_gate)
    inventory=base.pilot_gates(primary,variants)
    inventory.update(unit_scope_changed=True,includes_contiguous_excerpts=True,does_not_override_original_paragraph_gate=True)
    write_json(output/'expanded_scope_inventory.json',inventory)
    write_json(output/'repair_log.private.json',{'protocol':PROTOCOL,'raw_parent_records_preserved':True,'records':[r['provenance'] for r in results]})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'parent_package_manifest_sha256':file_sha(package/'manifest.json'),'parent_audit_manifest_sha256':file_sha(original_audit/'manifest.json'),
        'raw_response_hashes':coverage['response_hashes'],'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print('Recovered two explicit contiguous-excerpt programs; originals unchanged. Combined provisional set: 4 seed pairs + 2 related variants. No training.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--responses-dir',type=Path,required=True)
    p.add_argument('--original-audit',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    run(a.package,a.responses_dir,a.original_audit,a.output_dir)


if __name__=='__main__':main()
