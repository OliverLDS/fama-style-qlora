"""Freeze the twenty-paper roster after private opening metadata checks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalized(text):
    return re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', text).casefold())


def freeze(root, output):
    inputs = {}
    def load(relative):
        path = root/relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    base = load('docs/author_inventory_v2/inventory.json')
    plan = load('docs/author_inventory_v3/acquisition_plan.json')
    receipts = {r['paper_id']:r for r in load('research_runs/author_inventory_v3/acquisition_attempt_01.private.json')['files']}
    checks = {r['paper_id']:r for r in load('research_runs/author_inventory_v3/metadata_checks.private.json')['checks']}
    additions = []
    for planned in plan['papers']:
        pid = planned['paper_id']; receipt, check = receipts[pid], checks[pid]
        if receipt['status'] != 'downloaded' or not check['metadata_accepted']:
            raise ValueError('unverified_acquisition:'+pid)
        if hashlib.sha256((root/receipt['pdf']).read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('pdf_changed:'+pid)
        row = dict(planned, **check.get('metadata_overrides', {}))
        if row['version_year'] >= 2020 or (row['publication_year'] is not None and row['publication_year'] >= 2020):
            raise ValueError('cutoff_violation')
        authors = {normalized(a) for a in row['authors']}
        required = {'Fama_coauthors':{'eugeneffama'}, 'Shiller':{'robertjshiller'},
                    'Holmstrom':{'bengtholmstrom'}, 'Baker_Wurgler':{'malcolmbaker','jeffreywurgler'}}
        if not required[row['style_group']] <= authors:
            raise ValueError('author_group_mismatch')
        additions.append(dict(row, pdf=receipt['pdf'], pdf_sha256=receipt['sha256'],
            pdf_pages=check['pdf_pages'], existing_split='unassigned_new_acquisition',
            collection='author_expansion_v3_20261101', selected_pairs=0,
            quality='Opening bibliographic metadata checked; full prose not independently verified.',
            opening_metadata_check=check))
    papers = base['papers'] + additions
    for key in ('paper_id','pdf_sha256'):
        if len({p[key] for p in papers}) != len(papers):
            raise ValueError('duplicate_'+key)
    if len({normalized(p['title']) for p in papers}) != len(papers):
        raise ValueError('duplicate_work_title')
    roster = []
    for old in base['roster']:
        extra = [p for p in additions if p['style_group']==old['style_group']]
        ids = old['paper_ids'] + [p['paper_id'] for p in extra]
        if len(ids) != len(set(ids)) or len(ids) != 20:
            raise ValueError('roster_not_twenty')
        selected = [p for p in papers if p['paper_id'] in ids]
        roster.append(dict(old, paper_ids=ids, total_distinct_papers=20, new_this_expansion=len(extra),
            sole_author_papers=sum(len(p['authors'])==1 for p in selected),
            exact_author_list_variants=len({tuple(normalized(a) for a in p['authors']) for p in selected})))
    result = dict(base, protocol='author_inventory_v3', base_protocol=base['protocol'],
        date='2026-11-01', papers=papers, roster=roster, input_sha256=inputs,
        authors=sorted({a for p in papers for a in p['authors']}),
        summary={'registered_papers_total':len(papers), 'selected_roster_papers':80,
                 'new_papers_downloaded':len(additions), 'old_split_changes':0,
                 'new_training_pairs_created':0, 'model_calls':0, 'gpu_jobs':0},
        limitations=base['limitations']+[
            'Twenty distinct works per group is source coverage, not twenty paired or verified papers.',
            'New pairing roles are versioned separately; inventory preserves prior split labels.',
            'Related papers and survey overlap require passage checks before admission.',
            'Opening metadata checks do not establish transcription correctness or author-specific style.'])
    if output.exists():
        raise ValueError('refuse_overwrite_inventory')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze(args.root,args.output)
