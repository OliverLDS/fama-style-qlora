"""A teacher-only cohort must preserve targets, order and accepted Gemini inputs."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('audit_gemini500',ROOT/'scripts/data/audit_gemini_track_500.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Gemini500Tests(unittest.TestCase):
    def cohort(self):
        parent=[];new=[];groups=['Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler']
        for i in range(500):
            source=f'Old input {i}.';target=f'Human target paragraph {i}.';generator=m.MODEL if i>=286 else 'historical teacher'
            a=dict(id=str(i),source=source,plain_english=source,source_sha256=m.sha(source),target=target,original_paragraph=target,
                target_sha256=m.sha(target),parent_id='p'+str(i),parent_target_id='p'+str(i),paper_id='paper'+str(i%53),style_group=groups[i%4],authors=['Author'],
                split='training',training_ready=True,quality=dict(text_pair_status='automated_pilot_approved'),provenance=dict(generator=generator))
            b=copy.deepcopy(a);s=source if i>=286 else f'Gemini input {i}.'
            b.update(id=a['id']+'::gemini_track_v1',source=s,plain_english=s,source_sha256=m.sha(s),provenance=dict(generator=m.MODEL,teacher_track='Gemini',matched_mixed_id=a['id'],input_origin='approved_mixed_reuse' if i>=286 else 'new_Gemini_generation'))
            parent.append(a);new.append(b)
        return parent,new

    def test_target_identity_and_accepted_inputs_are_preserved(self):
        parent,new=self.cohort();a=m.validate(parent,new)
        self.assertEqual(a['accepted_Gemini_sources_reused_unchanged'],214)
        self.assertTrue(a['matched_human_targets_unchanged'])

    def test_changed_target_order_or_reused_input_fails_even_after_rehash(self):
        for issue in ['target','order','reused_source','teacher']:
            with self.subTest(issue=issue):
                parent,new=self.cohort()
                if issue=='target':new[0].update(target='Edited target',original_paragraph='Edited target',target_sha256=m.sha('Edited target'))
                elif issue=='order':new[0],new[1]=new[1],new[0]
                elif issue=='reused_source':new[-1].update(source='Changed',plain_english='Changed',source_sha256=m.sha('Changed'))
                else:new[0]['provenance']['generator']='other teacher'
                with self.assertRaises(ValueError):m.validate(parent,new)


if __name__=='__main__':unittest.main()
