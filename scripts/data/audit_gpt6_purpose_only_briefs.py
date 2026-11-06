"""Audit all 500 returned briefs without filtering or changing their outputs."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import statistics

SPEC = importlib.util.spec_from_file_location('purpose_collection', Path(__file__).with_name('prepare_gpt6_purpose_only_500.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
m = c.m
OUT = c.OUT / 'brief_audit_v1'

CONTENT_FOLLOWUPS = {
    'P139': 'The accepted target explicitly says bonds trade at a discount relative to stocks; the brief describes a liquidity premium and changed prices/returns without retaining that direction.',
    'P289': 'Permanent pool is stronger than the target history that these firms have never paid dividends; the target does not establish that they never will.',
    'P085': 'IPO terminology appears where the target discusses new lists; check whether the realized population is narrowed.',
    'P213': 'IPO terminology appears where the target discusses new lists; check whether the realized population is narrowed.',
    'P233': 'IPO supply shift may narrow a target covering exchange new lists, including non-IPO listings.',
    'P313': 'Check that an IPO-age argument is not presented as defining the entire new-list population.',
    'P329': 'One content note describes IPO firms where the main target population is new lists; preserve the distinction when assessing the output.',
    'P353': 'The topic describes IPO cohorts, while the target explicitly includes IPO and non-IPO new lists; check downstream population wording.'
}


def numerals(text):
    text = re.sub(r'@A\d+', '', text)
    return set(re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%)?', text))


def prepare():
    if OUT.exists():
        raise ValueError('refuse_overwrite_brief_audit')
    rows, files = c.load_all_briefs()
    targets = {r['id']: r for r in m.lines(c.OUT / 'targets_0500.private.jsonl')}
    diagnostics = []
    for row in rows:
        b = row['brief']
        text = ' '.join([b['topic']] + b['key_points'] + list(b['anchors'].values()))
        target = targets[row['id']]['target']
        metrics = m.contrast(text, target)
        added = sorted(numerals(text) - numerals(target))
        missing = sorted(numerals(target) - numerals(text))
        flags = []
        if added:
            flags.append('numeral_representation_difference')
        if len(missing) > 4:
            flags.append('multiple_numeral_differences')
        if re.search(r'\b(ambig|unclear|uncertain source|inconsistent|not supplied|unresolved|missing context|typo)', text, re.I):
            flags.append('uncertainty_word_candidate')
        if metrics['longest_ordinary_words'] >= 12:
            flags.append('long_ordinary_span')
        if len(text.split()) < .23*len(target.split()):
            flags.append('short_content_inventory')
        if row['id'] in CONTENT_FOLLOWUPS:
            flags.append('targeted_content_followup')
        diagnostics.append(dict(id=row['id'], target_sha256=row['target_sha256'],
            raw_brief_sha256=c.pilot.a.p.p3.v.r.b.b.row_hash(row),
            content_words=len(text.split()), target_words=len(target.split()),
            content_to_target_word_ratio=len(text.split())/len(target.split()),
            longest_ordinary_span=metrics['longest_ordinary_words'], relation_edges=len(b['relationships']),
            alphabetically_indexed=b['key_points'] == sorted(b['key_points'], key=lambda x: (x.casefold(), x)),
            added_numeral_strings=added, missing_numeral_strings=missing,
            advisory_flags=flags, content_followup=CONTENT_FOLLOWUPS.get(row['id']),
            disposition='retain_unchanged_for_reconstruction_with_advisories', training_ready=False))
    OUT.mkdir()
    (OUT / 'diagnostics.private.jsonl').write_text(m.jsonl(diagnostics))
    summary = dict(total_briefs=500, new_response_files=17, reused_pilot_briefs=5,
        statuses=dict(Counter(r['status'] for r in rows)),
        primary_functions=dict(Counter(r['brief']['purpose']['primary'] for r in rows)),
        empty_relation_lists=sum(not r['brief']['relationships'] for r in rows),
        nonempty_relation_lists=sum(bool(r['brief']['relationships']) for r in rows),
        median_content_to_target_word_ratio=statistics.median(r['content_to_target_word_ratio'] for r in diagnostics),
        alphabetically_indexed=sum(r['alphabetically_indexed'] for r in diagnostics),
        long_ordinary_span_flags=sum(r['longest_ordinary_span'] >= 12 for r in diagnostics),
        lexical_numeral_screen_flagged=sum(bool(set(r['advisory_flags']) - {'targeted_content_followup'}) for r in diagnostics),
        targeted_content_followup_ids=sorted(CONTENT_FOLLOWUPS),
        included_for_reconstruction=500, brief_edits=0, records_discarded=0,
        relation_annotation_status='sparse_explicit_graphs_not_proof_of_absent_semantic_relations',
        independent_semantic_verification=False, training_ready=False,
        method='All-record field/identity/reference/overlap/numeral screening, with targeted content checks; not exhaustive claim alignment.',
        limitations=['All ready statuses are model reports, not independent approval.',
            'Most numeral differences are representation changes, footnote omission, abbreviated dates or derivable quantities; flags are not error labels.',
            'The prompt permits empty relation lists when points already express the dependencies; do not claim a fully explicit relation corpus.',
            'Source ordering, long-span and compactness proxies do not establish style erasure or semantic equivalence.',
            'Content concerns remain visible for assessing reconstructed pairs; no automatic filtering or silent response repairs.'],
        provider_calls=0, training_calls=0)
    m.dump(OUT / 'summary.json', summary)
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)), raw_response_files=files,
        collection_manifest_sha256=m.digest(c.OUT / 'manifest.json'),
        pilot_briefs_sha256=m.digest(c.OUT / 'reused/pilot_briefs.private.jsonl'),
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    return check()


def check():
    rows, files = c.load_all_briefs()
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)) or manifest['raw_response_files'] != files or manifest['collection_manifest_sha256'] != m.digest(c.OUT / 'manifest.json') or manifest['pilot_briefs_sha256'] != m.digest(c.OUT / 'reused/pilot_briefs.private.jsonl'):
        raise ValueError('brief_audit_dependencies_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('brief_audit_changed')
    diagnostics = m.lines(OUT / 'diagnostics.private.jsonl')
    if [r['id'] for r in diagnostics] != [r['id'] for r in rows] or len(diagnostics) != 500:
        raise ValueError('audit_coverage')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
