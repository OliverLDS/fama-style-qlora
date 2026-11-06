"""Deterministic nested cohorts of approved, unique training targets; no ML."""
from collections import defaultdict
import hashlib

STAGES = (200,500,1000,1500,2000)
GROUPS = ('Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler')


def nested_cohorts(records, sizes=STAGES, groups=GROUPS, seed='data_scaling_v1', frozen_prefix=()):
    if not groups or len(set(groups))!=len(groups):raise ValueError('distinct_groups_required')
    if list(sizes)!=sorted(set(sizes)) or any(n<=0 or n%len(groups) for n in sizes):
        raise ValueError('sizes_must_increase_and_balance_groups')
    def rank(value): return hashlib.sha256((seed+'|'+value).encode()).hexdigest()
    records=list(records);parents={};targets={}
    for row in records:
        if row.get('split')!='training' or row.get('training_ready') is not True:
            raise ValueError('unapproved_or_nontraining_record')
        if row['style_group'] not in groups: raise ValueError('unexpected_author_group')
        parent=row.get('parent_target_id',row['paper_id']+'|'+row['paragraph_id'])
        target_hash=hashlib.sha256(row['target'].encode()).hexdigest()
        if target_hash in targets and targets[target_hash]!=parent:
            raise ValueError('duplicate_target_across_parents_requires_resolution')
        targets[target_hash]=parent
        if parent in parents:
            previous=parents[parent]
            if previous['target']!=row['target'] or previous['paper_id']!=row['paper_id'] or previous['style_group']!=row['style_group']:
                raise ValueError('conflicting_parent_versions')
            # Multiple neutral paraphrases do not increase the unique-target count.
            if rank(row['source'])<rank(previous['source']):parents[parent]=row
        else:parents[parent]=row
    frozen_parents=set();prefix_by_group=defaultdict(list)
    for i,row in enumerate(frozen_prefix):
        parent=row.get('parent_target_id',row['paper_id']+'|'+row['paragraph_id'])
        if parent in frozen_parents or row not in records or row['style_group']!=groups[i%len(groups)]:
            raise ValueError('changed_duplicate_or_unbalanced_frozen_prefix')
        frozen_parents.add(parent);prefix_by_group[row['style_group']].append(row)
        parents[parent]=row  # Retain the already frozen neutral-input variant.
    if len(frozen_prefix)%len(groups):raise ValueError('unbalanced_frozen_prefix')
    pools={}
    for group in groups:
        papers=defaultdict(list)
        for parent,row in parents.items():
            if row['style_group']==group and parent not in frozen_parents:
                papers[row['paper_id']].append((parent,row))
        for paper in papers:papers[paper].sort(key=lambda item:rank(item[0]))
        paper_order=sorted(papers,key=rank)
        ordered=[];depth=0
        while any(depth<len(papers[p]) for p in paper_order):
            ordered.extend(papers[p][depth][1] for p in paper_order if depth<len(papers[p]))
            depth+=1
        pools[group]=prefix_by_group[group]+ordered
    cohorts={}
    for size in sizes:
        need=size//len(groups)
        if any(len(pools[g])<need for g in groups):
            raise ValueError('insufficient_approved_unique_targets_for_stage:'+str(size))
        cohorts[size]=[row for i in range(need) for group in groups for row in [pools[group][i]]]
    return cohorts
