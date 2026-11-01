"""CPU integrity checks for training-only semantic anchors; no entailment oracle."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fama_style.research.teacher_pair_anchors import construct,validate_review,build,CONSTRAINTS
from fama_style.training.teacher_pair_pilot import sha


class SemanticAnchors(unittest.TestCase):
    def fixture(self,i=0,family='time'):
        prefix=f'Case {i}: '
        source=prefix+'The firm expanded before the change. This is a timing statement.'
        target=prefix+'Before the change, the firm expanded. This is a timing statement.'
        variant=prefix+'The timing statement reports firm expansion before the change.'
        original={'example_id':str(i),'paper_id':'paper'+str(i),'author_group_id':'group'+str(i),
            'authors':['Synthetic author'],'split':'training','source':source,'source_sha256':sha(source),
            'target':target,'target_sha256':sha(target),'annotation':{'alignment_status':'aligned_to_archived_text',
                'features':['scope_frequency','event_order_boundary','reported_numeric_values','condition_assumption']},
            'original_curation':{'original_needs_review':False}}
        error={'time':'ordering_reversal','scope':'scope_reversal','numbers':'magnitude_change','reasoning':'unsupported_reason_substitution'}[family]
        # These synthetic records test schema and exact edits, not error-type semantics.
        claims=[{'claim_id':'C'+str(j),'statement':statement,'actors':['firm'],'relation':'timing',
            'constraints':{k:[] for k in CONSTRAINTS},'source_evidence':evidence,'target_evidence':evidence,
            'variant_evidence':variant_evidence} for j,statement,evidence,variant_evidence in [
                (1,'Firm expansion precedes change.','before the change','before the change'),
                (2,'The statement describes timing.','timing statement','timing statement')]]
        draft={'example_id':str(i),'source_sha256':sha(source),'target_sha256':sha(target),'family':family,
            'faithful_variant':variant,'protected_names':[],'claims':claims,'negative_edit':{
                'before':'Before','after':'After','error_type':error,'affected_claim_ids':['C1'],
                'explanation':'Synthetic timing reversal used to test exact-edit reconstruction.'}}
        row=construct(draft,original)
        review={k:row[k] for k in ('example_id','variant_sha256','target_sha256','rejected_sha256')}
        review.update({'claim_checks':[{'claim_id':c['claim_id'],'preserved':True,'notes':'Synthetic binding check.'} for c in claims],
            'added_claim_identified':False,'negative_is_unfaithful':True,'review_notes':'Synthetic fixture, not semantic gold.',
            'claim_ledger_sha256':sha(json.dumps(row['claims'],sort_keys=True,ensure_ascii=False)),
            'negative_edit_sha256':sha(json.dumps(row['negative_edit'],sort_keys=True,ensure_ascii=False))})
        return original,draft,review

    def test_chosen_exact_and_rejected_one_contiguous_edit(self):
        original,draft,review=self.fixture();row=construct(draft,original);validate_review(row,review)
        self.assertEqual(row['chosen'],original['target'])
        self.assertEqual(row['rejected'],original['target'].replace('Before','After',1))
        self.assertFalse(row['training_ready']);self.assertFalse(row['semantic_gold_verified'])

    def test_changed_source_numeric_values_missing_claims_or_ambiguous_edit_rejected(self):
        original,draft,review=self.fixture()
        for field,value in [('split','development'),('source_sha256','wrong')]:
            bad=copy.deepcopy(original);bad[field]=value
            with self.assertRaises(ValueError):construct(draft,bad)
        for field,value in [('faithful_variant',draft['faithful_variant']+' 999'),('target_sha256','wrong')]:
            bad=copy.deepcopy(draft);bad[field]=value
            with self.assertRaises(ValueError):construct(bad,original)
        bad=copy.deepcopy(draft);bad['negative_edit']['before']='absent'
        with self.assertRaises(ValueError):construct(bad,original)
        bad=copy.deepcopy(draft);bad['negative_edit']['affected_claim_ids']=['unknown']
        with self.assertRaises(ValueError):construct(bad,original)
        bad=copy.deepcopy(draft);bad['claims'][0]['variant_evidence']='not present'
        with self.assertRaises(ValueError):construct(bad,original)

    def test_review_needs_same_texts_and_every_claim(self):
        original,draft,review=self.fixture();row=construct(draft,original)
        for field,value in [('variant_sha256','wrong'),('claim_checks',review['claim_checks'][:-1]),
                            ('negative_is_unfaithful',False),('added_claim_identified',True)]:
            bad=copy.deepcopy(review);bad[field]=value
            with self.assertRaises(ValueError):validate_review(row,bad)
        changed=copy.deepcopy(row);changed['claims'][0]['relation']='reversed_timing'
        with self.assertRaises(ValueError):validate_review(changed,review)
        changed=copy.deepcopy(row);changed['negative_edit']['affected_claim_ids']=['C2']
        with self.assertRaises(ValueError):validate_review(changed,review)

    def test_balanced_package_is_reproducible_and_never_opens_development(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);coverage=root/'coverage';coverage.mkdir()
            originals=[];drafts=[];reviews=[]
            for i in range(12):
                original,draft,review=self.fixture(i,('scope','time','numbers','reasoning')[i//3])
                originals.append(original);drafts.append(draft);reviews.append(review)
            path=coverage/'per_pair.private.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in originals))
            (coverage/'development.private.jsonl').write_text('Unreadable development bodies must not be parsed.')
            (coverage/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p.read_text()) for p in coverage.iterdir() if p.is_file()}}))
            draft_path=root/'drafts.jsonl';draft_path.write_text(''.join(json.dumps(r)+'\n' for r in drafts))
            review_path=root/'reviews.jsonl';review_path.write_text(''.join(json.dumps(r)+'\n' for r in reviews))
            a=build(coverage,draft_path,review_path,root/'a');b=build(coverage,draft_path,review_path,root/'b')
            self.assertEqual(a,b);self.assertEqual(a['development_records_read'],0)
            for file in (root/'a').iterdir():self.assertEqual(file.read_bytes(),(root/'b'/file.name).read_bytes())
            with self.assertRaises(ValueError):build(coverage,draft_path,review_path,root/'a')
            drafts[0]['family']='time';draft_path.write_text(''.join(json.dumps(r)+'\n' for r in drafts))
            with self.assertRaises(ValueError):build(coverage,draft_path,review_path,root/'invalid')


if __name__=='__main__':unittest.main()
