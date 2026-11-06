"""CPU-only audit of matched targets, reused sources and nested Gemini cohorts."""
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
MODEL='gemini-3-flash-preview'


def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(parent,cohort):
    if len(parent)!=500 or len(cohort)!=500:raise ValueError('five_hundred_pairs_required')
    if len({x['id'] for x in cohort})!=500 or len({x['parent_target_id'] for x in cohort})!=500:raise ValueError('duplicate_identity')
    reused=0;overlaps=[];origins=Counter()
    for a,b in zip(parent,cohort):
        if any(a[k]!=b[k] for k in ['target','original_paragraph','target_sha256','parent_id','parent_target_id','paper_id','style_group','authors']):
            raise ValueError('matched_target_or_order_changed')
        p=b['provenance']
        if b['id']!=a['id']+'::gemini_track_v1' or p['matched_mixed_id']!=a['id'] or p['generator']!=MODEL or p['teacher_track']!='Gemini':
            raise ValueError('teacher_or_identity_binding')
        if b['source']!=b['plain_english'] or b['source_sha256']!=sha(b['source']) or b['target_sha256']!=sha(b['target']):raise ValueError('text_hash')
        if b['split']!='training' or not b['training_ready'] or b['quality']['text_pair_status']!='automated_pilot_approved':raise ValueError('unapproved_pair')
        if a['provenance'].get('generator')==MODEL:
            if b['source']!=a['source'] or p['input_origin']!='approved_mixed_reuse':raise ValueError('accepted_Gemini_input_regenerated_or_changed')
            reused+=1
        origins[p['input_origin']]+=1
        overlaps.append(difflib.SequenceMatcher(None,re.findall(r'\w+',b['source'].casefold()),re.findall(r'\w+',b['target'].casefold()),autojunk=False).ratio())
    if reused!=214:raise ValueError('expected_214_accepted_reuses')
    return dict(pairs=500,teacher=MODEL,matched_human_targets_unchanged=True,accepted_Gemini_sources_reused_unchanged=reused,
        input_origins=dict(origins),groups=dict(Counter(x['style_group'] for x in cohort)),papers=len({x['paper_id'] for x in cohort}),
        median_source_target_word_similarity=statistics.median(overlaps),exact_whitespace_normalized_source_target_matches=sum(' '.join(x['source'].split())==' '.join(x['target'].split()) for x in cohort),
        training_calls=0)


def audit():
    root=ROOT/'data/data_scaling_v1/gemini_track_v1';parent=ROOT/'data/data_scaling_v1/stage_0500_expansion_v1/training.private.jsonl'
    for folder in [parent.parent,root/'stage_0200',root/'stage_0500']:
        for name,h in json.loads((folder/'manifest.json').read_text())['files'].items():
            if digest(folder/name)!=h:raise ValueError('release_file_changed')
    small=root/'stage_0200/training.private.jsonl';large=root/'stage_0500/training.private.jsonl'
    if not large.read_bytes().startswith(small.read_bytes()) or len(rows(small))!=200:raise ValueError('nested_prefix')
    result=validate(rows(parent),rows(large))
    if sorted(result['groups'].values())!=[125]*4 or sorted(Counter(x['style_group'] for x in rows(small)).values())!=[50]*4:raise ValueError('group_balance')
    result.update(nested_200_byte_prefix_unchanged=True,parent_mixed_sha256=digest(parent),stage_200_sha256=digest(small),stage_500_sha256=digest(large))
    return result


if __name__=='__main__':print(json.dumps(audit(),indent=2))
