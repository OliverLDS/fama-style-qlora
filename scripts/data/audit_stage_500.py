"""CPU-only final checks for the nested 500-pair release; no model imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata
import difflib

ROOT=Path(__file__).resolve().parents[2]


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def audit(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for rel,h in manifest['files'].items():
        if digest(folder/rel)!=h:raise ValueError('release_changed')
    path=folder/'training.private.jsonl';data=rows(path)
    prefix=ROOT/'data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl'
    if not path.read_bytes().startswith(prefix.read_bytes()) or manifest['parent_200_sha256']!=digest(prefix):
        raise ValueError('frozen_prefix_changed')
    expected={'Fama_coauthors':125,'Shiller':125,'Holmstrom':125,'Baker_Wurgler':125}
    if len(data)!=500 or dict(Counter(r['style_group'] for r in data))!=expected:raise ValueError('group_counts')
    for key in ['id','parent_target_id']:
        if len({r[key] for r in data})!=500:raise ValueError('duplicate_'+key)
    if len({norm(r['target']) for r in data})!=500:raise ValueError('duplicate_target')
    for r in data:
        if r['split']!='training' or not r['training_ready'] or r['quality']['text_pair_status']!='automated_pilot_approved':
            raise ValueError('unapproved_row')
        if any(r[field]!=r[alias] for field,alias in [('source','plain_english'),('target','original_paragraph')]):
            raise ValueError('alias_changed')
        if sha(r['source'])!=r['source_sha256'] or sha(r['target'])!=r['target_sha256']:raise ValueError('text_hash')
    # The same order contract can serve as the prefix for the later nested stages.
    groups=('Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler')
    if any(r['style_group']!=groups[i%4] for i,r in enumerate(data)):raise ValueError('nested_group_order')
    near=[];tokens=[re.findall(r'\w+',r['target'].casefold()) for r in data]
    for i,a in enumerate(tokens):
        for j,b in enumerate(tokens[:i]):
            if min(len(a),len(b))/max(len(a),len(b))<.74 or len(set(a)&set(b))/max(1,min(len(set(a)),len(set(b))))<.65:continue
            ratio=difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
            if ratio>=.85:near.append(dict(first=data[j]['id'],second=data[i]['id'],similarity=ratio))
    if near:raise ValueError('near_duplicate_targets_require_audit')
    matches=[]
    for r in data:
        a=re.findall(r'\w+',r['source'].casefold());b=re.findall(r'\w+',r['target'].casefold())
        matches.append(difflib.SequenceMatcher(None,a,b,autojunk=False).ratio())
    return dict(rows=500,unique_parents=500,groups=expected,original_200_byte_prefix_unchanged=True,
                unique_targets=500,near_duplicate_target_flags=0,papers=len({r['paper_id'] for r in data}),
                median_input_target_word_overlap=statistics.median(matches),
                exact_normalized_input_target_matches=sum(norm(r['source'])==norm(r['target']) for r in data),
                new_model_calls=0,training_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/data_scaling_v1/stage_0500_expansion_v1')
    print(json.dumps(audit(p.parse_args().data),indent=2))
