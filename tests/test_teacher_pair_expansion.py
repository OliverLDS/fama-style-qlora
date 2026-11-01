"""Data expansion preserves rows, flags and hold decisions without model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fama_style.research.teacher_pair_expansion import select,training_annotations,ADMIT
from fama_style.training.teacher_pair_pilot import sha,prompt


class TeacherPairExpansion(unittest.TestCase):
    def fixture(self):
        parent=[];annotations={}
        for i in range(3):
            source='Synthetic neutral input '+str(i);target='Synthetic author target '+str(i)
            row={'example_id':'T'+str(i),'paragraph_id':'P'+str(i),'paper_id':'paper'+str(i),
                 'authors':['Author '+str(i)],'author_group_id':'group'+str(i),'split':'training',
                 'source':source,'target':target,'source_sha256':sha(source),'target_sha256':sha(target),
                 'publication_year':2010,'needs_review':i>0,'needs_external_context':i>0}
            parent.append(row)
            annotations[row['paragraph_id']]={'plain_english':source,'original_paragraph':target,
                'source_quality':{'status':'uncertain','issues':['Unspecified earlier discussion.'],'needs_external_context':True},
                'review_notes':['Original flag retained.']}
        fields=('example_id','paper_id','authors','author_group_id','split','source','target','source_sha256','target_sha256')
        old={k:parent[0][k] for k in fields}|{'prompt':prompt(parent[0]['source']),'pilot_eligible':True}
        dev=old|{'example_id':'D','paper_id':'devpaper','authors':['Dev author'],'author_group_id':'devgroup',
                 'source':'Synthetic dev input','target':'Synthetic dev target','split':'development'}
        dev.update(source_sha256=sha(dev['source']),target_sha256=sha(dev['target']),prompt=prompt(dev['source']))
        previous=[old,dev]
        prior={r['example_id']:{'decision':'approve_limited_pilot' if i==0 else 'hold_unresolved_upstream_flags'} for i,r in enumerate(parent)}
        decisions={r['example_id']:{'decision':ADMIT if i==1 else 'hold_unbound_context',
                    'source_sha256':r['source_sha256'],'target_sha256':r['target_sha256'],
                    'alignment_notes':'Synthetic matching claims.','context_disposition':'Local claim complete; no context invented.'}
                   for i,r in enumerate(parent) if i>0}
        return parent,previous,prior,decisions,annotations

    def test_old_pairs_development_and_flags_are_preserved(self):
        args=self.fixture();before=copy.deepcopy(args)
        train,dev,audit=select(*args)
        self.assertEqual([r['example_id'] for r in train],['T0','T1'])
        self.assertEqual(train[0],args[1][0]);self.assertEqual(dev,[args[1][1]])
        self.assertEqual(args,before)
        added=next(r for r in audit if r['example_id']=='T1')
        self.assertTrue(added['original_needs_review']);self.assertTrue(added['original_needs_external_context'])
        self.assertEqual(added['upstream_source_quality']['issues'],['Unspecified earlier discussion.'])

    def test_every_previously_held_row_needs_a_bound_decision(self):
        args=list(self.fixture());del args[3]['T2']
        with self.assertRaises(ValueError):select(*args)
        args=list(self.fixture());args[3]['T1']['source_sha256']='changed'
        with self.assertRaises(ValueError):select(*args)
        args=list(self.fixture());args[3]['T1']['context_disposition']=''
        with self.assertRaises(ValueError):select(*args)

    def test_mismatched_upstream_or_cross_split_author_cannot_enter(self):
        args=list(self.fixture());args[4]['P1']['plain_english']='changed'
        with self.assertRaises(ValueError):select(*args)
        args=list(self.fixture());args[1][1]['authors']=args[0][1]['authors']
        with self.assertRaises(ValueError):select(*args)
        args=list(self.fixture());args[0][1]['publication_year']=2023
        with self.assertRaises(ValueError):select(*args)

    def test_nontraining_bodies_not_deserialized(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'audit.jsonl'
            path.write_text(json.dumps({'split':'training','paragraph_id':'P','annotation':{'safe':'synthetic'}})+'\n'
                '{"split":"reserved_test","annotation": INVALID_UNREADABLE_RESERVED}\n'
                '{"split":"development","annotation": INVALID_UNREADABLE_DEVELOPMENT}\n')
            self.assertEqual(training_annotations(path),{'P':{'safe':'synthetic'}})


if __name__=='__main__':unittest.main()
