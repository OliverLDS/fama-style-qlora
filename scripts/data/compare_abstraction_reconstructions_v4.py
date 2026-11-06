"""Compare ten returned pilot paragraphs; preserve all outputs without repair."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

SPEC = importlib.util.spec_from_file_location('reconstruction_packet', Path(__file__).with_name('prepare_abstraction_reconstruction_v4.py'))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
m = a.m
OUT = a.p.OUT / 'comparison_v1'

ASSESSMENTS = {
    'P250': dict(organization='B explicitly compares the city means and equal medians; A lists the same figures coherently.',
                 meaning='Both preserve the numbers and expectation/realization distinction; B adds a comparison directly derivable from those figures.',
                 overall='Selective guidance benefit; no established author-style benefit.'),
    'P264': dict(organization='Near tie. Both primarily follow the alphabetical data-variable inventory instead of the reference ordering.',
                 meaning='Both retain variable definitions, data-source roles, normalization and the closedness formula.',
                 overall='Little visible added value from the purpose label in this data-description case.'),
    'P291': dict(organization='Both connect liquidity, collateral, price discovery and disclosure coherently; B makes the chain more explicit.',
                 meaning='A changes least-cost to particularly low-cost. B retains the comparative claim and adds an explanatory stock-market contrast.',
                 overall='B preserves one comparison more precisely; A has substantially lower lexical sequence similarity to the target.'),
    'P458': dict(organization='B has a slightly clearer final inference; A repeats the point about unavailable quick profits.',
                 meaning='Both distinguish explanatory value from trading profits; B makes an implied evaluation conclusion explicit.',
                 overall='Small subjective organization advantage to B, with nearly identical target lexical similarity.'),
    'P484': dict(organization='Both return to the original broad claim, age evidence, coordinated trading, measurement sequence.',
                 meaning='B verbalizes a relationship-field caveat that age evidence does not directly compare retail and professional investors; the target does not state that critique.',
                 overall='A is leaner. B demonstrates annotation-commentary transfer, not a demonstrated improvement in style.')
}


def load():
    a.check()
    originals = m.lines(a.p.OUT / 'stage_1_brief_batches/inputs_01.jsonl')
    targets = {x['id']: x for x in originals}
    results, files = {}, {}
    validate = a.p.p3.v.r.old.validate_response
    for job in m.read(a.OUT / 'jobs.json'):
        path = a.p.OUT / 'responses/stage_2_rewrites' / job['response']
        rows = m.lines(path)
        if [r.get('id') for r in rows] != job['ids']:
            raise ValueError('returned_output_coverage')
        for row in rows:
            validate(row, targets[row['id']])
        results[job['condition']] = {r['id']: r for r in rows}
        files[str(path.relative_to(a.p.OUT))] = m.digest(path)
    return originals, results, files


def prepare():
    if OUT.exists():
        raise ValueError('refuse_overwrite_comparison')
    originals, results, files = load()
    pairs, candidates = [], []
    for t in originals:
        rid = t['id']
        row = dict(id=rid, target_sha256=t['target_sha256'], assessment=ASSESSMENTS[rid], metrics={})
        for condition, rows in results.items():
            raw = rows[rid]
            text = raw['plain_english']
            metrics = m.contrast(text, t['original_paragraph'])
            row['metrics'][condition] = dict(words=len(text.split()), sentences=len(m.sentences(text)), **metrics)
            candidates.append(dict(id=rid, condition=condition, split='training',
                source=text, source_sha256=m.sha(text), target=t['original_paragraph'], target_sha256=t['target_sha256'],
                raw_response=raw, disposition='retained_as_observed_no_edits', training_ready=False,
                independent_semantic_verification=False))
        row['between_output_word_similarity'] = m.contrast(results['content_only'][rid]['plain_english'],
            results['purpose_and_relations'][rid]['plain_english'])['word_similarity']
        pairs.append(row)
    OUT.mkdir()
    (OUT / 'comparisons.private.jsonl').write_text(m.jsonl(pairs))
    (OUT / 'candidates.private.jsonl').write_text(m.jsonl(candidates))
    summary = dict(cases=5, observed_outputs=10, output_edits=0,
        word_counts={c: sum(r['metrics'][c]['words'] for r in pairs) for c in results},
        median_word_similarity_to_target={c: statistics.median(r['metrics'][c]['word_similarity'] for r in pairs) for c in results},
        guided_longer_cases=sum(r['metrics']['purpose_and_relations']['words'] > r['metrics']['content_only']['words'] for r in pairs),
        guided_lower_word_similarity_cases=sum(r['metrics']['purpose_and_relations']['word_similarity'] < r['metrics']['content_only']['word_similarity'] for r in pairs),
        all_response_statuses_rewritten=all(r['rewrite_status'] == 'rewritten' for rows in results.values() for r in rows.values()),
        review_method='Unblinded Codex text assessment and surface metrics; no independent semantic/style judge.',
        guidance_commentary_transfer_ids=['P484'], less_precise_comparative_ids_content_only=['P291'],
        recommendation='Content-only is the provisional neutral-prose default; retain purpose/relationships for the brief route and selective research.',
        no_general_guidance_or_author_style_advantage_established=True, training_ready=False,
        requested_model='GPT-6', requested_effort='High', actual_settings_independently_verified=False,
        provider_calls=0, training_calls=0, detector_calls=0,
        limitations=['Five training-side cases, three author groups, no Fama case, one generation per condition.',
            'UI model/decoding/session independence are not verified; sampling variation is not separated from guidance effects.',
            'Sequence similarity is a surface diagnostic, sensitive to names and fixed terminology; it is not an author-style or semantic score.',
            'The guided metadata can add explicit implications or critique beyond the common content, as observed for P484.'])
    m.dump(OUT / 'summary.json', summary)
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)), response_files=files,
        packet_manifest_sha256=m.digest(a.OUT / 'manifest.json'),
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    return check()


def check():
    originals, results, files = load()
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)) or manifest['response_files'] != files or manifest['packet_manifest_sha256'] != m.digest(a.OUT / 'manifest.json'):
        raise ValueError('comparison_dependencies_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('comparison_artifact_changed')
    targets = {r['id']: r for r in originals}
    candidates = m.lines(OUT / 'candidates.private.jsonl')
    if len(candidates) != 10 or len({(r['id'], r['condition']) for r in candidates}) != 10:
        raise ValueError('candidate_coverage')
    for row in candidates:
        raw = results[row['condition']][row['id']]
        if row['raw_response'] != raw or row['source'] != raw['plain_english'] or row['source_sha256'] != m.sha(row['source']) or row['target'] != targets[row['id']]['original_paragraph'] or row['training_ready']:
            raise ValueError('unmodified_output_binding')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
