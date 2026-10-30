"""Freeze the 13-paper author expansion from verified local acquisition metadata.

Offline metadata only: no paragraph extraction, model calls, split reassignment
or reserved text access. Keeps the v1 snapshot and its acquisition files intact.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', value).casefold())


def expand(root):
    sources = {}
    def load(relative):
        path = root / relative
        sources[relative] = sha(path)
        return json.loads(path.read_text())

    base = load('docs/author_inventory_v1/inventory.json')
    plan = load('docs/author_inventory_v2/acquisition_plan.json')
    receipt = load('archive/author_expansion_v2_20261030/acquisition_attempt_01.private.json')
    checks = load('research_runs/author_inventory_v2/metadata_checks.private.json')
    by_id = {r['paper_id']: r for r in receipt['files']}
    checked = {r['paper_id']: r for r in checks['checks']}
    papers = list(base['papers'])
    for row in plan['papers']:
        item = by_id[row['paper_id']]
        check = checked[row['paper_id']]
        assert item['status'] == 'downloaded', 'incomplete_acquisition'
        assert sha(root / item['pdf']) == item['sha256'] == check['pdf_sha256'], 'pdf_changed'
        assert all(check[key] for key in ('title_verified', 'all_author_surnames_present', 'version_year_visible')), 'unverified_opening_metadata'
        assert row['version_year'] < plan['cutoff_exclusive_year'], 'version_after_cutoff'
        assert row['publication_year'] < plan['cutoff_exclusive_year'], 'publication_after_cutoff'
        assert check['verified_version_year'] == row['version_year'], 'unverified_version_date'
        names = {normalized(a) for a in row['authors']}
        if row['style_group'] == 'Baker_Wurgler':
            assert {'malcolmbaker', 'jeffreywurgler'} <= names, 'both_joint_authors_required'
        elif row['style_group'] == 'Holmstrom':
            assert 'bengtholmstrom' in names, 'missing_author'
        elif row['style_group'] == 'Shiller':
            assert 'robertjshiller' in names, 'missing_author'
        else:
            raise ValueError('unknown_expansion_group')
        papers.append(row | {'pdf': item['pdf'], 'pdf_sha256': item['sha256'],
                             'pdf_pages': check['pdf_pages'], 'opening_metadata_check': check,
                             'existing_split': 'unassigned_new_acquisition', 'collection': 'author_expansion_v2_20261030',
                             'selected_pairs': 0, 'quality': 'container/opening metadata checked; no full prose extraction, annotation or training-readiness claim'})
    assert len({p['paper_id'] for p in papers}) == len(papers), 'duplicate_paper_id'
    assert len({p['pdf_sha256'] for p in papers}) == len(papers), 'identical_pdf_counted_twice'
    assert len({normalized(p['title']) for p in papers}) == len(papers), 'same_title_counted_twice'

    roster = []
    for previous in base['roster']:
        group = previous['style_group']
        ids = previous['paper_ids'] + [p['paper_id'] for p in plan['papers'] if p['style_group'] == group]
        assert len(ids) == len(set(ids)) == plan['target_distinct_papers_per_group'], 'roster_count_not_13'
        selected = [p for p in papers if p['paper_id'] in ids]
        exact = Counter(tuple(normalized(a) for a in p['authors']) for p in selected)
        roster.append({'style_group': group, 'label': {'Fama_coauthors': 'Fama/coauthors', 'Holmstrom': 'Holmstrom/coauthors',
                                                     'Shiller': 'Robert J. Shiller', 'Baker_Wurgler': 'Baker-Wurgler/coauthors'}[group],
                       'paper_ids': ids, 'total_distinct_papers': len(ids),
                       'new_this_expansion': sum(p['paper_id'] not in previous['paper_ids'] for p in selected),
                       'exact_author_list_variants': len(exact),
                       'sole_author_papers': sum(len(p['authors']) == 1 for p in selected),
                       'split_policy': 'all existing roles preserved; no new assignments made'})
    # Author participation remains metadata, not a claim of identifiable prose contribution.
    authors = {}
    for p in papers:
        for name in p['authors']:
            row = authors.setdefault(normalized(name), {'name': name, 'paper_ids': []})
            row['paper_ids'].append(p['paper_id'])
    registered = {p['pdf']: p['pdf_sha256'] for p in papers}
    for p in papers:
        assert sha(root / p['pdf']) == p['pdf_sha256'], 'previously_registered_pdf_changed'
    for artifact in base['other_pdf_artifacts']:
        assert sha(root / artifact['pdf']) == artifact['sha256'], 'previous_artifact_changed'
    actual = {str(p.relative_to(root)) for folder in ('archive', 'research_runs')
              for p in (root / folder).rglob('*.pdf')}
    known = set(registered) | {p['pdf'] for p in base['other_pdf_artifacts']}
    assert actual == known, 'unregistered_pdf_requires_explicit_inventory'
    return {'protocol': plan['protocol'], 'date': plan['date'], 'base_protocol': base['protocol'],
            'papers': papers, 'roster': roster, 'authors': sorted(authors.values(), key=lambda p: normalized(p['name'])),
            'other_pdf_artifacts': base['other_pdf_artifacts'], 'input_sha256': sources,
            'summary': {'registered_papers_before': len(base['papers']), 'new_papers_downloaded': len(plan['papers']),
                        'registered_papers_total': len(papers), 'unique_authors_total': len(authors),
                        'selected_roster_papers': sum(r['total_distinct_papers'] for r in roster),
                        'pdf_files_total': len(actual), 'manual_downloads_needed': 0,
                        'old_split_changes': 0, 'reserved_paragraph_bodies_read': False,
                        'new_training_pairs_created': 0, 'gpu_jobs': 0, 'model_calls': 0},
            'limitations': base['limitations'] + [
                'Thirteen papers per group is a source-availability target, not balanced usable paragraph counts.',
                'Both Baker and Wurgler occur on all group papers; four added works include other coauthors.',
                'Holmstrom group includes coauthors; full author identities retained. Shiller main texts are sole-authored.',
                'Exact PDF/title duplication screened; passage overlap and related-version review still required before splitting.',
                'Discussion comments, forewords and other authors in PDF containers are excluded from intended prose attribution.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = expand(args.root)
    if args.output.exists():
        raise ValueError('refuse_to_overwrite_inventory_snapshot')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
