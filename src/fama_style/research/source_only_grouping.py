"""CPU-only, paper-held-out source-boundary retrieval; no generation or adapters."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import statistics

from .author_control_replication import verify_files
from .author_comparison_corpus import file_sha, words
from .human_prose_review import normalized
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='source_only_grouping_v1'
METHODS=('keep','fixed_pairs','retrieval_k3')
FEATURES=('sentence_count','boundary_position','left_length','right_length','word_overlap',
          'right_reference','right_reason','right_example','right_contrast','right_addition',
          'right_condition','left_semicolon')
CONFIG={'protocol':PROTOCOL,'neighbors':3,'one_neighbor_per_paper':True,'distance':'mean_absolute_fixed_scale',
        'merge_vote_threshold':2/3,'features':list(FEATURES),'sentence_count_scale':10,'word_length_scale':80,
        'folds':'leave_one_whole_paper_out','methods':list(METHODS),'tuning':'none',
        'split':'training_only','max_partition_solutions_retained':2}


def source_view(row):
    """Prediction whitelist. Original text, target labels and claim IDs stay out."""
    a=row['annotation'];units=[s['exact_text'] for s in a['source_schema']['sentences']]
    if not units or any(not s.strip() for s in units) or normalized(' '.join(units))!=normalized(a['plain_english']):
        raise ValueError('source_sentence_coverage')
    return {'paragraph_id':row['paragraph_id'],'paper_id':row['paper_id'],'sentences':units}


def partitions(source,target):
    """Find up to two contiguous source-unit partitions with the target claim sets."""
    source=tuple(frozenset(g) for g in source);target=tuple(frozenset(g) for g in target)
    if not source or not target or any(not g for g in source+target):return []
    @lru_cache(None)
    def visit(i,j):
        if j==len(target):return ((),) if i==len(source) else ()
        found=[];union=set()
        for stop in range(i+1,len(source)+1):
            union.update(source[stop-1])
            if not union<=target[j]:break
            if union==target[j]:
                for tail in visit(stop,j+1):
                    found.append((stop,)+tail)
                    if len(found)==2:return tuple(found)
        return tuple(found)
    return list(visit(0,0))


def audit_labels(row):
    if not row['schema_valid'] or row['audit']['errors'] or row['audit']['program_errors']:
        return {'status':'mechanically_invalid'}
    a=row['annotation'];claims=a['semantic_ledger']['claims']
    if any(c['alignment_status']!='preserved' for c in claims):return {'status':'uncertain_claim_alignment'}
    source=[u['claim_ids'] for u in a['source_schema']['sentences']]
    target=[u['claim_ids'] for u in a['target_schema']['sentences']]
    universe={c['claim_id'] for c in claims}
    for plan in (source,target):
        if not plan or any(not g or len(g)!=len(set(g)) or set(g)-universe for g in plan):
            return {'status':'invalid_claim_groups'}
        if set().union(*(set(g) for g in plan))!=universe:return {'status':'incomplete_claim_coverage'}
    solutions=partitions(source,target)
    if not solutions:return {'status':'split_required' if len(target)>len(source) else 'not_adjacent_merge_representable'}
    if len(solutions)>1:return {'status':'ambiguous_partition'}
    ends=set(solutions[0]);n=len(source)
    labels=[int(i not in ends) for i in range(1,n)]
    return {'status':'eligible','source_sentences':n,'target_sentences':len(target),'merge_labels':labels,
            'source_claim_repetition':len([c for g in source for c in g])>len(universe),
            'has_merge':any(labels),'needs_external_context':a['source_quality']['needs_external_context'],
            'annotator_needs_review':a['needs_review']}


def collect(records):
    audit=[];eligible=[]
    for row in records:
        if row['split']!='training':continue
        result=audit_labels(row)
        audit.append({'paragraph_id':row['paragraph_id'],'paper_id':row['paper_id'],**result})
        if result['status']=='eligible':eligible.append({'source':source_view(row),'labels':result['merge_labels']})
    if len({r['paragraph_id'] for r in audit})!=len(audit):raise ValueError('duplicate_training_id')
    return audit,eligible


def boundary_features(view):
    units=view['sentences'];n=len(units);result=[]
    def begins(text,pattern):return int(bool(re.match(pattern,text.strip(),re.I)))
    for i,(left,right) in enumerate(zip(units,units[1:])):
        a,b=words(left.lower()),words(right.lower());aa,bb=set(a),set(b)
        result.append([min(n/10,1),(i+1)/max(1,n-1),min(len(a)/80,1),min(len(b)/80,1),
            len(aa&bb)/max(1,len(aa|bb)),
            begins(right,r'(?:this|these|that|those|it|its|they|their|such)\b'),
            begins(right,r'(?:because|thus|therefore|consequently|hence|so)\b'),
            begins(right,r'(?:for example|for instance|in other words)\b'),
            begins(right,r'(?:but|however|yet|nevertheless|by contrast|in contrast)\b'),
            begins(right,r'(?:also|moreover|in addition|furthermore)\b'),
            begins(right,r'(?:if|when|unless|provided that)\b'),int(';' in left)])
    return result


def library(examples,heldout_paper):
    result=[]
    for example in examples:
        view=example['source']
        if view['paper_id']==heldout_paper:continue
        for i,(features,label) in enumerate(zip(boundary_features(view),example['labels'])):
            result.append({'paper_id':view['paper_id'],'paragraph_id':view['paragraph_id'],
                           'boundary':i,'features':features,'label':label})
    return result


def retrieve(features,pool,query_paper):
    # Defensive exclusion also applies if a caller accidentally passes a full pool.
    candidates=[(statistics.mean(abs(a-b) for a,b in zip(features,r['features'])),r)
                for r in pool if r['paper_id']!=query_paper]
    candidates.sort(key=lambda x:(x[0],x[1]['paper_id'],x[1]['paragraph_id'],x[1]['boundary']))
    neighbors=[];papers=set()
    for distance,row in candidates:
        if row['paper_id'] in papers:continue
        papers.add(row['paper_id']);neighbors.append({k:row[k] for k in ('paper_id','paragraph_id','boundary','label')}|{'distance':distance})
        if len(neighbors)==3:break
    if len(neighbors)!=3:raise ValueError('three_other_papers_required')
    probability=sum(r['label'] for r in neighbors)/3
    return int(probability>=CONFIG['merge_vote_threshold']),neighbors


def predict(view,pool,method):
    features=boundary_features(view)
    if method=='keep':return [0]*len(features),[]
    if method=='fixed_pairs':return [int(i%2==0) for i in range(len(features))],[]
    if method!='retrieval_k3':raise ValueError('unknown_method')
    predictions=[retrieve(f,pool,view['paper_id']) for f in features]
    return [p[0] for p in predictions],[p[1] for p in predictions]


def plan_from_labels(view,labels):
    if len(labels)!=len(view['sentences'])-1 or any(type(x) is not int or x not in (0,1) for x in labels):
        raise ValueError('boundary_labels')
    groups=[[1]]
    for i,label in enumerate(labels,2):
        if label:groups[-1].append(i)
        else:groups.append([i])
    return groups


def score(rows):
    tp=fp=fn=tn=0
    for r in rows:
        for gold,pred in zip(r['gold_labels'],r['predicted_labels']):
            tp+=gold==pred==1;tn+=gold==pred==0;fp+=gold==0 and pred==1;fn+=gold==1 and pred==0
    changed=[r for r in rows if any(r['gold_labels'])];identity=[r for r in rows if not any(r['gold_labels'])]
    exact=sum(r['gold_labels']==r['predicted_labels'] for r in rows)
    papers=sorted({r['paper_id'] for r in rows})
    return {'rows':len(rows),'papers':len(papers),'boundaries':tp+fp+fn+tn,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'merge_precision':tp/(tp+fp) if tp+fp else None,'merge_recall':tp/(tp+fn) if tp+fn else None,
        'merge_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'boundary_accuracy':(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None,
        'exact_paragraphs':exact,'exact_paragraph_rate':exact/len(rows) if rows else None,
        'macro_paper_exact_rate':statistics.mean(sum(r['gold_labels']==r['predicted_labels'] for r in rows if r['paper_id']==p)/sum(r['paper_id']==p for r in rows) for p in papers) if papers else None,
        'merge_rows':len(changed),'merge_rows_exact':sum(r['gold_labels']==r['predicted_labels'] for r in changed),
        'identity_rows':len(identity),'identity_rows_exact':sum(r['gold_labels']==r['predicted_labels'] for r in identity)}


def evaluate(examples):
    rows=[]
    for example in examples:
        view=example['source'];pool=library(examples,view['paper_id'])
        for method in METHODS:
            prediction,neighbors=predict(view,pool,method)
            rows.append({'paragraph_id':view['paragraph_id'],'paper_id':view['paper_id'],'method':method,
                'gold_labels':example['labels'],'predicted_labels':prediction,
                'predicted_source_sentence_groups':plan_from_labels(view,prediction),'neighbors':neighbors})
    return rows


def run(audit_directory,output):
    verify_files(audit_directory)
    records=[json.loads(l) for l in (audit_directory/'audits.private.jsonl').read_text().splitlines()]
    audited,examples=collect(records)
    if len(audited)!=60 or len({r['paper_id'] for r in audited})!=12:raise ValueError('expected_60_training_rows_12_papers')
    predictions=evaluate(examples)
    summary={'protocol':PROTOCOL,'status':'cpu_evaluation_complete_no_generation',
        'config':CONFIG,'audited_training_rows':len(audited),'audit_statuses':dict(Counter(r['status'] for r in audited)),
        'eligible_rows':len(examples),'eligible_papers':len({e['source']['paper_id'] for e in examples}),
        'eligible_single_sentence_rows':sum(len(e['source']['sentences'])==1 for e in examples),
        'eligible_rows_with_external_context_flag':sum(r.get('needs_external_context',False) for r in audited if r['status']=='eligible'),
        'eligible_rows_flagged_by_annotator':sum(r.get('annotator_needs_review',False) for r in audited if r['status']=='eligible'),
        'eligible_rows_with_repeated_source_claim':sum(r.get('source_claim_repetition',False) for r in audited),
        'methods':{method:score([r for r in predictions if r['method']==method]) for method in METHODS},
        'per_paper':{p:{method:score([r for r in predictions if r['paper_id']==p and r['method']==method]) for method in METHODS}
            for p in sorted({r['paper_id'] for r in predictions})},
        'prediction_inputs':'source sentence text plus paper ID for exclusion only; no query reference, claim IDs or taxonomy labels',
        'annotation_context':'Source sentences and gold alignments came from joint paired GPT-6 annotations; not independent extraction.',
        'heldout_unit':'paper; authors can recur in other training papers','tuning_performed':False,
        'source_projection_verified_by_tests':True,'style_quality_evaluated':False,'semantic_equivalence_certified':False,
        'development_rows_scored':0,'reserved_rows_used':0,'original_fama_splits_used':0,
        'external_model_calls':0,'gpu_jobs_started':0,'neural_or_adapter_training_called':False}
    fresh(output);write_json(output/'config.json',CONFIG);write_json(output/'summary.json',summary)
    write_lines(output/'alignment_audit.private.jsonl',audited);write_lines(output/'predictions.private.jsonl',predictions)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'audit_manifest_sha256':file_sha(audit_directory/'manifest.json'),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s=run(a.audit,a.output);print(json.dumps({k:s[k] for k in ('audit_statuses','eligible_rows','methods')},indent=2))


if __name__=='__main__':main()
