import tests
import copy
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from fama_style.evaluation import source_control_review as review


@tests.requires_local_artifacts(review.ROOT / "archive/corpus_v1_pilot_version_1/comparison.json", review.ROOT / "data/fama_corpus_v1/curated.jsonl")
class SourceControlReviewTests(unittest.TestCase):
    def setUp(self):
        self.comparison = review.load_comparison(review.ROOT / 'archive/corpus_v1_pilot_version_1/comparison.json')
        self.bundle, self.mapping, self.scores, self.manifest = review.make_review(self.comparison)

    def completed_scores(self):
        scores = copy.deepcopy(self.scores)
        scores['reviewer'] = 'Synthetic test fixture; not an assessment'
        for row in scores['ratings']:
            row.update(claims=4, economic_precision=2, analytical_structure=2, concision=2)
        return scores

    def test_all_three_frozen_candidates_balanced_and_no_test_rows(self):
        counts = {slot: Counter() for slot in 'ABC'}
        source = {row['id']: row for row in self.comparison['results']}
        for example in self.bundle['examples']:
            row = source[example['id']]
            for candidate in example['candidates']:
                label = self.mapping['assignments'][example['id']][candidate['slot']]
                counts[candidate['slot']][label] += 1
                expected = {'unchanged_input':row['source'],'base':row['baseline_output'],'epoch_1':row['output']}
                self.assertEqual(candidate['text'],expected[label])
        self.assertEqual(len(source),21)
        self.assertEqual(len(self.scores['ratings']),63)
        for slot in 'ABC':
            self.assertEqual(dict(counts[slot]), {label:7 for label in review.LABELS})
        self.assertTrue(all(row['claims'] is None for row in self.scores['ratings']))
        self.assertNotIn('assignments',self.bundle)
        self.assertEqual(review.make_review(self.comparison)[0],self.bundle)

    def test_rejects_altered_inputs_mapping_and_wrong_score_bundle(self):
        review.validate_bundle(self.bundle,self.mapping,self.manifest)
        changed = copy.deepcopy(self.mapping)
        paragraph = next(iter(changed['assignments']))
        changed['assignments'][paragraph]['A']='wrong'
        with self.assertRaises(ValueError):
            review.validate_bundle(self.bundle,changed,self.manifest)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'changed.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                review.load_comparison(path)
        scores=self.completed_scores();scores['review_id']='wrong'
        with self.assertRaises(ValueError):
            review.analyze(self.bundle,self.mapping,scores)

    def test_no_summary_for_incomplete_duplicate_or_out_of_range_scores(self):
        with self.assertRaises(ValueError):
            review.analyze(self.bundle,self.mapping,self.scores)
        for value in (None,True,4.5,5,-1):
            scores=self.completed_scores();scores['ratings'][0]['claims']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                review.analyze(self.bundle,self.mapping,scores)
        scores=self.completed_scores();scores['ratings'][-1]=scores['ratings'][0]
        with self.assertRaises(ValueError):
            review.analyze(self.bundle,self.mapping,scores)

    def test_identical_source_and_pilot_candidates_must_tie(self):
        scores=self.completed_scores()
        for example in self.bundle['examples']:
            candidates=example['candidates']
            match=next(((a,b) for i,a in enumerate(candidates) for b in candidates[i+1:]
                        if review.normalized(a['text'])==review.normalized(b['text'])),None)
            if match:
                row=next(row for row in scores['ratings'] if row['id']==example['id'] and row['slot']==match[0]['slot'])
                row['concision']=1
                break
        else:
            self.fail('Expected the six known source-copy pairs')
        with self.assertRaises(ValueError):
            review.analyze(self.bundle,self.mapping,scores)

    def test_analysis_keeps_criteria_separate_and_counts_ties(self):
        scores=self.completed_scores()
        for row in scores['ratings']:
            if self.mapping['assignments'][row['id']][row['slot']]=='base':
                row.update(claims=2,claim_notes='Synthetic test only',concision=0)
        summary=review.analyze(self.bundle,self.mapping,scores)
        self.assertEqual(summary['identical_candidate_pairs'],6)
        self.assertEqual(summary['pairwise']['epoch_1_vs_unchanged_input']['claims'],{'wins':0,'ties':21,'losses':0})
        self.assertEqual(summary['pairwise']['epoch_1_vs_base']['claims'],{'wins':21,'ties':0,'losses':0})
        self.assertNotIn('combined_score',summary)

    def test_build_preserves_work_and_does_not_embed_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'review'
            path=review.ROOT/'archive/corpus_v1_pilot_version_1/comparison.json'
            review.build(path,out)
            html=(out/'review.html').read_text()
            self.assertNotIn('__REVIEW_PAYLOAD__',html)
            self.assertNotIn('"assignments"',html)
            self.assertNotIn('"epoch_1"',html)
            self.assertIn('connect-src \'none\'',html)
            self.assertNotIn('<script src=',html)
            (out/'blank_scores.json').write_text('user draft')
            with self.assertRaises(ValueError):
                review.build(path,out)
            self.assertEqual((out/'blank_scores.json').read_text(),'user draft')

    @unittest.skipUnless(shutil.which('node'),'Node is only used for local UI behavior checks')
    def test_offline_javascript_import_saving_and_duplicate_tie_behavior(self):
        template=Path(review.__file__).with_name('templates')/'source_control_review.html'
        html=template.read_text()
        script=re.findall(r'<script>\s*([\s\S]*?)</script>',html)[0]
        payload=json.dumps({'bundle':self.bundle,'blankScores':self.scores})
        harness=r'''
const assert=require('assert'), vm=require('vm');
const elements=new Map(),storage=new Map();
class Element{
 constructor(){this.children=[];this.value='';this.textContent='';}
 set id(value){this._id=value;elements.set(value,this);} get id(){return this._id;}
 append(...values){this.children.push(...values);} replaceChildren(){this.children=[];}
 setAttribute(){} click(){}
}
const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement(){return new Element();},activeElement:null};
document.getElementById('payload').textContent=PAYLOAD;
const context=vm.createContext({document,localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value)},console,Blob,URL,setTimeout,confirm:()=>true});
vm.runInContext(SCRIPT,context);
vm.runInContext(`
if(scores.ratings.length!==63)throw Error('Wrong number of candidate forms');
let target=bundle.examples.find(e=>e.candidates.some((a,i)=>e.candidates.slice(i+1).some(b=>norm(a.text)===norm(b.text))));
index=bundle.examples.indexOf(target);render();
let pair=target.candidates.filter(a=>target.candidates.some(b=>a.slot!==b.slot&&norm(a.text)===norm(b.text)));
applyRating(target.id,pair[0].slot,'concision',1);
if(rowFor(target.id,pair[1].slot).concision!==1)throw Error('Identical text ratings were not synchronized');
let imported=JSON.parse(JSON.stringify(scores));validateDraft(imported);
imported.review_id='wrong';let rejected=false;try{validateDraft(imported);}catch(e){rejected=true;}if(!rejected)throw Error('Accepted wrong review');
let draft=JSON.parse(localStorage.getItem(storageKey));if(draft.ratings.length!==63)throw Error('Autosave failed');
`,context);
'''
        harness=harness.replace('PAYLOAD',json.dumps(payload)).replace('SCRIPT',json.dumps(script))
        result=subprocess.run(['node','-e',harness],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
