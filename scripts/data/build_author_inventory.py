"""Inventory local paper coverage without inspecting reserved paragraph bodies.

Offline, standard-library only. Bibliography and counts may be public; paragraph
text, PDFs, receipts and extraction samples remain private. No data splits change.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name_key(name):
    return ''.join(c for c in unicodedata.normalize('NFKD', name.casefold())
                   if not unicodedata.combining(c))


def author_index(papers):
    authors = {}
    for paper in papers:
        for name in paper['authors']:
            item = authors.setdefault(name_key(name), {'name': name, 'papers': [], 'splits': {}})
            item['papers'].append(paper['paper_id'])
            item['splits'][paper['paper_id']] = paper['existing_split']
    return sorted(authors.values(), key=lambda row: name_key(row['name']))


def build(root):
    sources = {}
    def load(relative):
        path = root / relative
        sources[relative] = digest(path)
        return read(path)

    plan = load('docs/author_inventory_v1/acquisition_plan.json')
    split = load('data/fama_corpus_v1/split_manifest.json')['paper_assignments']
    archive = load('archive/fama_corpus_20261019/manifest.json')['files']
    papers = []
    # The archive manifest supplies file identity. Only nonreserved source records
    # are opened for bibliography/counts. Reserved FF2014 uses explicit metadata.
    fama = defaultdict(dict)
    for row in archive:
        fama[row['paper']][row['kind']] = row
    for paper_id, files in sorted(fama.items()):
        assignment = split[paper_id]
        pdf = root / 'archive/fama_corpus_20261019' / files['pdf']['archive_path']
        if assignment['split'] == 'test':
            metadata = {'title': 'A Five-Factor Asset Pricing Model',
                        'authors': ['Eugene F. Fama', 'Kenneth R. French'], 'year': 2014}
            raw_counts = None
        else:
            path = root / 'archive/fama_corpus_20261019' / files['dataset']['archive_path']
            sources[str(path.relative_to(root))] = digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            metadata = rows[0]
            raw_counts = dict(Counter(row.get('paragraph_type', 'unspecified') for row in rows))
        assert digest(pdf) == files['pdf']['sha256'], 'archived_pdf_changed'
        papers.append({'paper_id': paper_id, 'title': metadata['title'], 'authors': metadata['authors'],
                       'version_year': metadata['year'], 'publication_year': None,
                       'bibliographic_basis': 'existing archive; version year is not inferred publication year',
                       'existing_split': assignment['split'], 'collection': 'fama_corpus_v1',
                       'pdf': str(pdf.relative_to(root)), 'pdf_sha256': digest(pdf),
                       'selected_pairs': assignment['selected_rows'], 'raw_record_type_counts': raw_counts,
                       'reserved_counts_basis': 'split manifest only' if raw_counts is None else None})

    registry = load('research_runs/human_prose_annotation_v1/source_registry.private.json')
    selections = load('research_runs/human_prose_annotation_v1/package_v2/PRIVATE_SELECTION_AND_SPLITS.json')
    selected = Counter(r['paper_id'] for r in selections)
    # These files contain only training/development records, never reserved rows.
    milestone = {}
    for filename in ('train.provisional.private.jsonl', 'development.private.jsonl', 'quarantine.private.jsonl'):
        relative = 'research_runs/multi_author_milestone_v1/package_v1/datasets/' + filename
        path = root / relative
        sources[relative] = digest(path)
        counts = Counter()
        review = Counter()
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row['split'] not in ('training', 'development'):
                    raise ValueError('reserved_pair_in_nonreserved_inventory_input')
                counts[row['paper_id']] += 1
                review[row['paper_id']] += bool(row.get('needs_review'))
        milestone[filename] = (counts, review)
    for row in registry['papers']:
        pdf = root / row['pdf']
        assert digest(pdf) == row['pdf_sha256'], 'registered_pdf_changed'
        counts = {key: value[0][row['paper_id']] for key, value in milestone.items()}
        papers.append({key: row[key] for key in ('paper_id', 'title', 'authors', 'version_year', 'publication_year',
                                                'pdf', 'pdf_sha256', 'publication_evidence_url', 'download_url')} |
                      {'existing_split': row['split'], 'collection': 'human_prose_annotation_v1',
                       'selected_pairs': selected[row['paper_id']], 'milestone_counts': counts,
                       'provisional_training_needs_review': milestone['train.provisional.private.jsonl'][1][row['paper_id']],
                       'quality': 'selected/annotated is not verified meaning-equivalent or training-ready; flags retained'})
    relation = load('research_runs/relation_pilot_v1/source_registry.private.json')
    for row in relation['papers']:
        assert digest(root / row['pdf']) == row['pdf_sha256'], 'relation_pdf_changed'
        papers.append({key: row[key] for key in ('paper_id', 'title', 'authors', 'version_year', 'publication_year',
                                                'pdf', 'pdf_sha256', 'publication_evidence_url', 'download_url')} |
                      {'existing_split': 'discovery_pilot_nonreserved', 'collection': 'relation_pilot_v1',
                       'selected_pairs': None, 'quality': 'earlier structural annotation exposure; not part of the frozen 100 pairs'})

    receipt = load('archive/author_expansion_20261030/acquisition_receipt.private.json')
    by_id = {r['paper_id']: r for r in receipt['files']}
    inspection = load('research_runs/author_inventory_v1/pdf_metadata.private.json')
    inspected = {r['paper_id']: r for r in inspection}
    checks = load('research_runs/author_inventory_v1/metadata_checks.private.json')
    checked = {r['paper_id']: r for r in checks['checks']}
    for row in plan['papers']:
        acquisition = by_id[row['paper_id']]
        assert acquisition['status'] in ('downloaded', 'already_present'), 'incomplete_acquisition'
        assert digest(root / acquisition['pdf']) == acquisition['sha256'], 'new_pdf_changed'
        check = checked[row['paper_id']]
        assert check['pdf_sha256'] == acquisition['sha256'], 'inspection_pdf_mismatch'
        assert all(check[key] for key in ('title_matches_normalized', 'all_author_surnames_present', 'version_year_visible')), 'opening_metadata_not_confirmed'
        assert row['version_year'] < plan['cutoff_exclusive_year'] and row['publication_year'] < plan['cutoff_exclusive_year']
        papers.append(row | {'existing_split': 'unassigned_new_acquisition', 'collection': 'author_expansion_20261030',
                             'pdf': acquisition['pdf'], 'pdf_sha256': acquisition['sha256'], 'selected_pairs': 0,
                             'pdf_pages': inspected[row['paper_id']]['pdf_page_count'],
                             'opening_metadata_check': check,
                             'quality': 'download/container/title checked only; no prose extraction or annotation yet'})

    known = {p['pdf'] for p in papers}
    unassigned = []
    excluded = {'R_O1959.pdf': 'previously rejected: zero eligible blocks under frozen extraction filters',
                'R_GH1983.pdf': 'previously rejected: only three eligible blocks',
                'R_R1973.pdf': 'previously rejected: only two eligible blocks',
                'R_H1988.pdf': 'previously excluded book excerpt: includes Levmore text; original journal PDF substituted'}
    for parent in ('archive', 'research_runs'):
        for pdf in sorted((root / parent).rglob('*.pdf')):
            relative = str(pdf.relative_to(root))
            if relative not in known:
                unassigned.append({'pdf': relative, 'sha256': digest(pdf),
                                   'status': excluded.get(pdf.name, 'nonpaper_evidence' if 'detector_results' in relative else 'unregistered_requires_identification')})
    assert len({p['paper_id'] for p in papers}) == len(papers), 'duplicate_paper_id'
    authors = author_index(papers)
    existing = [p for p in papers if p['collection'] != 'author_expansion_20261030']
    roster = []
    for group in plan['roster']:
        before = [p for p in existing if (group['style_group'] == 'Fama_coauthors' and p['collection'] == 'fama_corpus_v1')
                  or p['paper_id'] in group.get('existing_paper_ids', [])]
        added = [p for p in papers if p.get('style_group') == group['style_group']]
        roster.append(group | {'existing_papers': len(before), 'new_papers_downloaded': len(added),
                               'total_papers': len(before) + len(added),
                               'paper_ids': [p['paper_id'] for p in before + added],
                               'split_policy': 'preserve all old roles; new paper roles unassigned pending extraction, overlap checks and a separate author-conditioned protocol'})
    return {'protocol': 'author_inventory_v1', 'date': plan['date'], 'papers': papers,
            'authors': authors, 'roster': roster, 'other_pdf_artifacts': unassigned,
            'summary': {'existing_registered_papers': len(existing), 'new_papers_downloaded': len(papers) - len(existing),
                        'registered_papers_total': len(papers), 'existing_unique_authors': len(author_index(existing)),
                        'unique_authors_total': len(authors), 'pdf_files_total': len(papers) + len(unassigned),
                        'other_pdf_artifacts': len(unassigned), 'manual_downloads_needed': 0,
                        'reserved_paragraph_bodies_read': False, 'old_split_changes': 0,
                        'new_annotations_or_training_pairs_created': 0, 'model_calls': 0, 'gpu_jobs': 0},
            'input_sha256': sources,
            'limitations': ['Bibliographic inventory, not a new curated dataset.',
                           'Author participation does not identify individual contribution to coauthored prose.',
                           'Historical publication does not rule out foundation-model pretraining exposure.',
                           'Public availability is not a redistribution or model-weight publication license.',
                           'Fama version dates retained separately; publication years not newly verified.',
                           'New-paper quality and cross-paper duplication remain unassessed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    if args.output.exists():
        raise ValueError('refuse_to_overwrite_inventory_snapshot')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
