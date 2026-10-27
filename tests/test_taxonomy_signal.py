"""Training-only structural evidence, annotation variation and grouping guards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import taxonomy_signal as t


class TaxonomySignalTests(unittest.TestCase):
    def test_identical_text_can_have_different_observations_without_true_change(self):
        a=[{'level':'sentence','canonical_id':'S001'}]
        b=[{'level':'sentence','canonical_id':'S002'}]
        d=t.observation_difference(a,b)
        self.assertEqual(d['sentence']['jaccard'],0)
        self.assertFalse(d['sentence']['absence_verified'])
        self.assertIsNone(d['paragraph']['jaccard'])
        self.assertEqual(t.structural_record('Firms invest.','Firms invest.')['sentence_delta'],0)

    def test_legacy_parenthesis_splitter_disagreement_is_not_selected_as_a_merge(self):
        source='Firms invest. Banks lend. Investors save.'
        target='Firms invest. (Banks lend.) Investors save.'
        stats=t.structural_record(source,target)
        self.assertLess(stats['sentence_delta'],0)  # Known legacy splitter limitation.
        row={'id':'synthetic','paper_key':'p','annotation_sentence_counts':{'source':3,'target':3},**stats}
        self.assertEqual(t.grouping_candidates([row]),[])

    def test_ranking_selects_one_per_paper_but_does_not_claim_semantic_readiness(self):
        rows=[{'id':ident,'paper_key':'p','source_sentence_count':n,'target_sentence_count':m,
               'annotation_sentence_counts':{'source':n,'target':m},'sentence_delta':m-n,'lexical_similarity':.5}
              for ident,n,m in [('one',3,2),('two',5,2),('same',2,2)]]
        chosen=t.grouping_candidates(rows)
        self.assertEqual([r['id'] for r in chosen],['two'])
        self.assertFalse(chosen[0]['training_ready']);self.assertFalse(chosen[0]['execution_ready'])

    def test_paper_probe_does_not_fit_on_its_own_held_paper(self):
        rows=[{'split':'train','paper_key':p,'source_sentence_count':1,'target_sentence_count':count,'source_words':20}
              for p,count in [('a',1),('b',2)]]
        probe=t.probe_counts(rows)
        self.assertEqual(probe['methods']['fixed_prior']['exact_count_accuracy'],0)
        self.assertEqual(probe['methods']['conditional']['exact_count_accuracy'],0)
        self.assertEqual(probe['methods']['identity']['exact_count_accuracy'],.5)
        self.assertFalse(probe['deployment_mapper_saved'])
        rows[0]['split']='validation'
        with self.assertRaises(ValueError):t.probe_counts(rows)

    def test_changed_audited_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input.json';path.write_text('{}')
            manifest=Path(folder)/'manifest.json';manifest.write_text(json.dumps({'files':{path.name:t.file_sha(path)}}))
            t.checked_manifest_file(path,manifest)
            path.write_text('{"changed": true}')
            with self.assertRaises(ValueError):t.checked_manifest_file(path,manifest)

    def test_grouping_is_hash_bound_ordered_and_does_not_license_new_relations(self):
        source='Firms invest. Banks lend. Investors save.';target='Firms invest and banks lend. Investors save.'
        args=(source,target,[[1,2],[3]],t.sha(source),t.sha(target))
        result=t.validate_grouping(*args)
        self.assertTrue(result['coverage_checked'])
        self.assertFalse(result['semantic_equivalence_verified']);self.assertFalse(result['new_semantic_relations_authorized'])
        for groups in ([[1,1],[2,3]],[[2,1],[3]],[[1],[3]],[[1,2,3]],[[True,2],[3]]):
            with self.assertRaises(ValueError):t.validate_grouping(source,target,groups,t.sha(source),t.sha(target))
        with self.assertRaises(ValueError):t.validate_grouping(source+' Changed.',*args[1:])


if __name__=='__main__':unittest.main()
