"""Training-only audit rejects changed evidence, split contamination and bad labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fama_style.research.teacher_pair_coverage import analyze,build
from fama_style.training.teacher_pair_pilot import sha,prompt


class TeacherPairCoverage(unittest.TestCase):
    def fixture(self):
        records=[];annotations=[]
        for i in range(46):
            source=f'Only {i+1} firms were observed before the change.'
            target=f'Before the change, only {i+1} firms were observed.'
            row={'example_id':str(i),'split':'training','paper_id':'paper'+str(i%3),
                'author_group_id':'group'+str(i%3),'authors':['Synthetic author'],
                'source':source,'target':target,'source_sha256':sha(source),'target_sha256':sha(target),
                'prompt':prompt(source),'pilot_eligible':True}
            records.append(row)
            annotations.append({'example_id':str(i),'source_sha256':sha(source),'target_sha256':sha(target),
                'features':['scope_frequency','event_order_boundary','reported_numeric_values'],
                'alignment_status':'aligned_to_archived_text','alignment_notes':'Synthetic aligned pair.',
                'coverage_notes':'Synthetic count, time and scope constraints.','major_claim_reordering':False})
        return records,annotations

    def test_descriptive_counts_are_not_a_learning_claim(self):
        records,annotations=self.fixture();results,summary=analyze(records,annotations)
        self.assertEqual(summary['coverage_families']['numeric_examples']['pairs'],46)
        self.assertEqual(summary['feature_coverage']['probability_outcome_tuples']['pairs'],0)
        self.assertEqual(summary['major_claim_order_changes_in_automated_review'],0)
        self.assertFalse(summary['training_authorized'])
        self.assertEqual(len(results),46)

    def test_incomplete_changed_or_nontraining_evidence_is_rejected(self):
        records,annotations=self.fixture()
        with self.assertRaises(ValueError):analyze(records,annotations[:-1])
        for field,value in [('split','development'),('source_sha256','changed'),('prompt','changed')]:
            changed=copy.deepcopy(records);changed[0][field]=value
            with self.assertRaises(ValueError):analyze(changed,annotations)
        changed=copy.deepcopy(annotations);changed[0]['target_sha256']='changed'
        with self.assertRaises(ValueError):analyze(records,changed)
        changed=copy.deepcopy(annotations);changed[-1]=changed[0]
        with self.assertRaises(ValueError):analyze(records,changed)

    def test_unbound_categories_and_unexplained_alignment_limits_are_rejected(self):
        records,annotations=self.fixture()
        for features in (['unknown'],['scope_frequency','scope_frequency'],['linked_numeric_comparisons']):
            changed=copy.deepcopy(annotations);changed[0]['features']=features
            with self.assertRaises(ValueError):analyze(records,changed)
        changed=copy.deepcopy(annotations);changed[0]['alignment_status']='qualified_alignment';changed[0]['alignment_notes']=''
        with self.assertRaises(ValueError):analyze(records,changed)

    def test_builder_never_parses_development_or_unmatched_audit_bodies(self):
        records,annotations=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);curated=root/'curated';(curated/'private_key').mkdir(parents=True)
            (curated/'training.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (curated/'development.private.jsonl').write_text('Not JSON. Never parse development bodies.')
            audit=[r|{'decision':{'decision':'carry_forward_v1' if i<16 else 'admit_limited_pilot'},'original_needs_review':i>=16} for i,r in enumerate(records)]
            (curated/'private_key/training_audit.private.jsonl').write_text('Unreadable unmatched body\n'+''.join(json.dumps(r)+'\n' for r in audit))
            manifest={'files':{str(p.relative_to(curated)):sha(p.read_text()) for p in curated.rglob('*') if p.is_file()}}
            (curated/'manifest.json').write_text(json.dumps(manifest))
            path=root/'annotations.private.jsonl';path.write_text(''.join(json.dumps(a)+'\n' for a in annotations))
            first=build(curated,path,root/'first');second=build(curated,path,root/'second')
            self.assertEqual(first,second)
            self.assertEqual(first['carried_forward_v1_training_pairs'],16)
            self.assertEqual(first['development_records_deserialized'],0)
            for p in (root/'first').iterdir():self.assertEqual(p.read_bytes(),(root/'second'/p.name).read_bytes())
            with self.assertRaises(ValueError):build(curated,path,root/'first')


if __name__=='__main__':unittest.main()
