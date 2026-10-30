"""Frozen hypotheses, author disjointness and fixed-anchor aggregation checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.research import author_control_replication as replication
from fama_style.research import author_comparison_corpus as corpus
from fama_style.research import author_comparison_diagnostics as diagnostics


class ReplicationTests(unittest.TestCase):
    def registries(self):
        policy = {'word_range': [25, 220]}
        old = {'excluded_author_surnames': ['fama', 'french', 'jensen', 'blume'], 'paragraph_match_policy': policy,
               'papers': [{'paper_id': 'T'+str(i), 'label': 1, 'topic': str(i), 'authors': ['Eugene Fama']} for i in range(7)] +
                         [{'paper_id': 'C'+str(i), 'label': 0, 'topic': str(i), 'authors': ['Old Author'+chr(97+i)]} for i in range(7)]}
        new = {'protocol': replication.PROTOCOL, 'matching_policy': policy,
               'controls': [{'paper_id': 'N'+str(i), 'matched_target': 'T'+str(i), 'topic': str(i), 'label': 0,
                             'authors': ['New Writer'+chr(97+i)], 'columns': 1, 'ocr_pages': [2, 3]} for i in range(7)]}
        return old, new

    def test_disjoint_new_authors_and_complete_target_coverage(self):
        old, new = self.registries()
        replication.validate_controls(new, old)
        for name in ('Kenneth French', 'Old Authora'):
            changed = copy.deepcopy(new); changed['controls'][0]['authors'] = [name]
            with self.assertRaisesRegex(ValueError, 'excluded_old_or_project_author'):
                replication.validate_controls(changed, old)
        new['controls'][0]['matched_target'] = 'T1'
        with self.assertRaisesRegex(ValueError, 'target_coverage'):
            replication.validate_controls(new, old)

    def test_policy_changes_rejected_and_accents_normalized(self):
        old, new = self.registries(); new['matching_policy'] = {'word_range': [10, 300]}
        with self.assertRaisesRegex(ValueError, 'quality_matching_policy_changed'):
            replication.validate_controls(new, old)
        self.assertEqual(replication.folded('Holmström'), 'holmstrom')

    def test_feature_hash_and_hypotheses_frozen(self):
        p = {'protocol': replication.PROTOCOL, 'feature_extractor_sha256': corpus.file_sha(Path(diagnostics.__file__)),
             'hypotheses': [{'feature': f, 'pattern': r, 'expected_target_minus_control': 'negative'}
                            for f, r in zip(replication.FEATURES, replication.PATTERNS)]}
        replication.verify_prereg(p, Path(diagnostics.__file__))
        p['hypotheses'][0]['expected_target_minus_control'] = 'positive'
        with self.assertRaisesRegex(ValueError, 'hypotheses_changed'):
            replication.verify_prereg(p, Path(diagnostics.__file__))
        p['feature_extractor_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'feature_extractor_changed'):
            replication.verify_prereg(p, Path(diagnostics.__file__))

    def test_new_names_do_not_change_frozen_mask(self):
        old, new = self.registries(); names = replication.frozen_names(old)
        self.assertIn('fama', names); self.assertNotIn('new', names)
        self.assertEqual(names, replication.frozen_names(copy.deepcopy(old)))

    def test_duplicate_screen_does_not_mutate_anchors(self):
        anchors = [{'id': 'a', 'text_sha256': 'one'}]
        rows = [{'id': 'b', 'text_sha256': 'one'}, {'id': 'c', 'text_sha256': 'two'},
                {'id': 'd', 'text_sha256': 'two'}, {'id': 'e', 'text_sha256': 'three'}]
        pool, rejected = replication.remove_duplicates(rows, anchors)
        self.assertEqual([r['id'] for r in pool], ['e']); self.assertEqual(len(rejected), 3)
        self.assertEqual(anchors, [{'id': 'a', 'text_sha256': 'one'}])

    def feature_rows(self):
        rows, matches = [], []
        for i in range(7):
            for suffix, group, value in [('a', 'target', 0), ('b', 'target', 4), ('o', 'old_control', 4), ('n', 'new_control', 3)]:
                rows.append({'id': str(i)+suffix, 'topic': str(i), 'group': group,
                             'features': {f: value for f in replication.FEATURES}})
            matches.append({'target_id': str(i)+'a'})
        return rows, matches

    def test_primary_uses_all_anchors_and_sensitivity_matched_subset(self):
        rows, matches = self.feature_rows()
        result = replication.summarize_features(rows, matches)
        for f in replication.FEATURES:
            self.assertEqual(result[f]['new_control_negative_topics'], 7)
            self.assertEqual(result[f]['new_control_median_difference'], -1)
            self.assertEqual(result[f]['matched_subset_median_difference'], -3)
            self.assertTrue(result[f]['meets_preregistered_direction_criterion'])
            self.assertEqual(result[f]['per_topic'][0]['target_n'], 2)
            self.assertEqual(result[f]['per_topic'][0]['matched_target_n'], 1)

    def test_ties_are_not_negative_and_threshold_requires_six_topics(self):
        rows, matches = self.feature_rows()
        for r in rows:
            if r['group'] == 'new_control' and r['topic'] in ('0', '1'):
                r['features'] = {f: 2 for f in replication.FEATURES}
        for result in replication.summarize_features(rows, matches).values():
            self.assertEqual(result['new_control_negative_topics'], 5)
            self.assertFalse(result['meets_preregistered_direction_criterion'])

    def test_manifest_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); (p/'x').write_text('original')
            (p/'manifest.json').write_text(json.dumps({'files': {'x': corpus.file_sha(p/'x')}}))
            replication.verify_files(p); (p/'x').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen_file_changed'):
                replication.verify_files(p)

    @unittest.skipUnless((corpus.ROOT/'data/author_control_replication_v1/manifest.json').exists(), 'Private controls unavailable')
    def test_private_anchor_equality_and_analysis_reproduction(self):
        directory = corpus.ROOT/'data/author_control_replication_v1'
        anchors = corpus.read_lines(directory/'original_target_anchors.private.jsonl')
        original = corpus.read_lines(corpus.ROOT/'data/author_comparison_v1/paragraphs.private.jsonl')
        self.assertEqual(anchors, [r for r in original if r['label'] == 1])
        replication.verify_files(directory)
        with tempfile.TemporaryDirectory() as temp:
            replication.analyze(directory, Path(temp)/'analysis')
            release = corpus.ROOT/'research_runs/author_control_replication_v1/analysis_v1'
            if release.exists():
                for name in ('features.private.jsonl', 'summary.json'):
                    self.assertEqual((Path(temp)/'analysis'/name).read_bytes(), (release/name).read_bytes())


if __name__ == '__main__':
    unittest.main()
