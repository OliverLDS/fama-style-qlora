"""CPU guards for dataset growth, source/target direction and complete batch coverage."""
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('expand500',ROOT/'scripts/data/expand_stage_500.py')
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)


class Expand500Test(unittest.TestCase):
    def test_id_repair_is_hash_bound_and_raw_response_is_unchanged(self):
        saved=module.RUN
        try:
            with tempfile.TemporaryDirectory() as folder:
                module.RUN=Path(folder);p=module.RUN/'response.json'
                module.write(p,dict(text='[{"id":"short","plain_english":"Same wording."}]'))
                before=p.read_bytes()
                module.write_lines(module.RUN/'response_id_repairs.private.jsonl',[
                    dict(response_relative_path=p.name,response_sha256=module.digest(p),from_id='short',to_id='complete')])
                parsed=module.parse_array(module.response_text(p),['complete'])
                self.assertEqual(parsed[0]['plain_english'],'Same wording.')
                self.assertEqual(p.read_bytes(),before)
                module.write(p,dict(text='[{"id":"short","plain_english":"Changed wording."}]'))
                with self.assertRaisesRegex(ValueError,'repair_binding'):module.response_text(p)
        finally:module.RUN=saved

    def test_pdf_target_repair_preserves_frozen_extraction(self):
        saved=module.RUN
        try:
            with tempfile.TemporaryDirectory() as folder:
                module.RUN=Path(folder);e=module.RUN/'evidence';e.write_bytes(b'evidence fixture')
                p=module.RUN/'missing_inputs.private.jsonl'
                module.write_lines(p,[dict(id='x',target='Citation ~1999!',target_sha256=module.sha('Citation ~1999!'))])
                before=p.read_bytes()
                module.write_lines(module.RUN/'target_repairs.private.jsonl',[
                    dict(id='x',before_sha256=module.sha('Citation ~1999!'),after='Citation (1999)',
                         after_sha256=module.sha('Citation (1999)'),evidence={'evidence':module.digest(e)},reason='PDF check')])
                result=module.effective_targets()[0]
                self.assertEqual(result['target'],'Citation (1999)')
                self.assertEqual(p.read_bytes(),before)
                e.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'evidence_changed'):module.effective_targets()
        finally:module.RUN=saved

    def test_complete_ids_required_even_when_model_reports_success(self):
        good=[dict(id='one',plain_english='Faithful source.'),dict(id='two',plain_english='Other source.')]
        self.assertEqual(module.parse_array('```json\n'+json.dumps(good)+'\n```',['one','two']),good)
        for bad in [good[:1],[good[0],good[0]],[good[0],dict(id='other',plain_english='Other')]]:
            with self.assertRaises(ValueError):module.parse_array(json.dumps(bad),['one','two'])

    def test_paper_interleaving_and_order_are_deterministic(self):
        rows=[dict(id=f'{p}{i}',paper_id=p) for p in ['A','B','C'] for i in range(4)]
        a=module.interleave(rows);self.assertEqual(a,module.interleave(list(reversed(rows))))
        self.assertEqual(len({r['paper_id'] for r in a[:3]}),3)

    def test_normalized_pair_keeps_human_target_and_provenance(self):
        pair=module.normalize_pair(dict(id='v2',parent_id='parent',paper_id='paper'),
            'Generated neutral input.','Original human paragraph.',dict(generator='teacher'),'automated review')
        self.assertEqual(pair['target'],'Original human paragraph.')
        self.assertEqual(pair['source'],'Generated neutral input.')
        self.assertEqual(pair['parent_target_id'],'parent')
        self.assertFalse(pair['quality']['independent_human_verification'])
        self.assertEqual(pair['target_sha256'],module.sha(pair['target']))


if __name__=='__main__':unittest.main()
