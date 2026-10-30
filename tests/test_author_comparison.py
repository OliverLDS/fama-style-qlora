"""Corpus quality, matching and paper-held-out diagnostics, without ML dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import author_comparison_corpus as corpus
from fama_style.research import author_comparison_diagnostics as diagnostics


class CorpusTests(unittest.TestCase):
    def test_logged_dictionary_join_and_ambiguous_compound(self):
        text,events,unresolved=corpus.normalize_candidate('We study manage-\nment and chang-\ning risk-\nneutral pay.',{'management','change'})
        self.assertIn('management',text);self.assertIn('changing',text)
        self.assertEqual(len(events),2);self.assertEqual(len(unresolved),1)
        self.assertIn('risk- neutral',text)
        for event in events:self.assertEqual(event['before'],('We study manage-\nment and chang-\ning risk-\nneutral pay.')[event['raw_start']:event['raw_end']])

    def test_incomplete_low_confidence_and_metadata_not_admitted(self):
        policy={'word_range':[3,80]}
        for text in ('these statements are incomplete.','This paragraph is not finished','Copyright rights are reserved here.'):
            reason,_=corpus.quality_reason({'raw_ocr':text,'min_ocr_confidence':1},policy)
            self.assertIsNotNone(reason)
        reason,_=corpus.quality_reason({'raw_ocr':'This complete statement contains enough ordinary words.','min_ocr_confidence':.4},policy)
        self.assertEqual(reason,'low_line_confidence')

    def test_matching_no_replacement_and_ignores_label(self):
        a=[{'id':'a','word_count':100,'window_position':.2,'label':1},{'id':'b','word_count':90,'window_position':.3,'label':1}]
        b=[{'id':'c','word_count':99,'window_position':.2,'label':0},{'id':'d','word_count':92,'window_position':.3,'label':0}]
        policy={'max_word_ratio':1.2,'max_window_position_difference':.55,'max_pairs_per_topic':2,'min_pairs_per_topic':2}
        first=corpus.choose_matches(a,b,policy);self.assertEqual(len({r['id'] for x,y,_ in first for r in (x,y)}),4)
        for r in a+b:r['label']=1-r['label']
        second=corpus.choose_matches(a,b,policy)
        self.assertEqual([(x['id'],y['id']) for x,y,_ in first],[(x['id'],y['id']) for x,y,_ in second])
        b[0]['word_count']=300
        with self.assertRaisesRegex(ValueError,'insufficient_matched'):corpus.choose_matches(a,b,policy)

    def test_excluded_coauthor_control_rejected(self):
        papers=[];pairs=[]
        for i,p in enumerate(sorted(corpus.TARGETS)):
            topic=f'topic{i}';control=f'C{i}'
            papers.extend([{'paper_id':p,'label':1,'authors':['Eugene Fama'],'ocr_pages':[2],'columns':1,'topic':topic},
                           {'paper_id':control,'label':0,'authors':['Independent Writer'],'ocr_pages':[2],'columns':1,'topic':topic}])
            pairs.append({'target_paper':p,'control_paper':control,'topic':topic})
        registry={'protocol':corpus.PROTOCOL,'papers':papers,'topic_pairs':pairs,'excluded_author_surnames':['fama','french','jensen','blume']}
        corpus.validate_registry(registry);registry['papers'][1]['authors']=['Kenneth French']
        with self.assertRaisesRegex(ValueError,'excluded_author'):corpus.validate_registry(registry)

    @unittest.skipUnless((corpus.ROOT/'data/author_comparison_v1/manifest.json').exists(),'Private comparison corpus unavailable')
    def test_private_build_reproduction_and_balanced_matches(self):
        root=corpus.ROOT/'research_runs/author_comparison_v1';release=corpus.ROOT/'data/author_comparison_v1'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'corpus';corpus.build(root/'registry.private.json',root/'acquisition_v1/ocr_release.private.jsonl',output)
            manifest=json.loads((release/'manifest.json').read_text())
            for name,digest in manifest['files'].items():self.assertEqual(corpus.file_sha(output/name),digest)
            rows=corpus.read_lines(output/'paragraphs.private.jsonl');by_match={}
            for r in rows:by_match.setdefault(r['match_id'],[]).append(r)
            self.assertEqual(len(rows),146);self.assertEqual(len(by_match),73)
            for rs in by_match.values():self.assertEqual({r['label'] for r in rs},{0,1})
            self.assertTrue(all(not r['training_ready'] for r in rows))


class DiagnosticTests(unittest.TestCase):
    def rows(self):
        return [{'id':str(i),'topic':topic,'label':label,'paper_id':topic+str(label),'text_sha256':str(i)}
                for i,(topic,label) in enumerate([('a',0),('a',1),('b',0),('b',1)])]

    def test_whole_topic_and_paper_holdout(self):
        rows=self.rows();self.assertEqual(len(list(diagnostics.folds(rows))),2)
        rows[2]['paper_id']=rows[0]['paper_id']
        with self.assertRaisesRegex(ValueError,'paper_leakage'):list(diagnostics.folds(rows))
        rows=self.rows();rows[2]['text_sha256']=rows[0]['text_sha256']
        with self.assertRaisesRegex(ValueError,'duplicate_text_leakage'):list(diagnostics.folds(rows))

    def test_train_only_scaling_and_finite_unseen_prediction(self):
        model=diagnostics.fit_ridge([[0],[2]],[0,1],[1,1],steps=50)
        self.assertEqual(model['means'],[1]);self.assertEqual(model['scales'],[1])
        before=copy.deepcopy(model);score=diagnostics.predict(model,[1000000])
        self.assertTrue(0<score<1);self.assertEqual(before,model)

    def test_names_and_citation_years_masked_and_vocabulary_train_only(self):
        row={'text':'Fama and French (2001) study dividends and payers.'}
        tokens=diagnostics.content_tokens(row,{'fama','french'})
        self.assertNotIn('fama',tokens);self.assertNotIn('french',tokens)
        training=[dict(row,label=0,paper_id='a'),{'text':'Other investors study prices.','label':1,'paper_id':'b'}]
        model=diagnostics.lexical_fit(training,{'fama','french'})
        self.assertNotIn('heldoutword',model)
        self.assertEqual(diagnostics.lexical_predict(model,{'text':'heldoutword'},{'fama'}),.5)

    def test_rhetorical_features_do_not_read_label_or_paper_identity(self):
        row={'text':'If costs rise, outcomes may change. Thus we report uncertainty.','word_count':12,'mean_ocr_confidence':1,
             'normalization_events':[],'window_position':.2,'version_year':2000,'columns':1,'authors':['A Writer'],
             'quality_flags':{'unrecognized_lowercase_words':[]},'label':0,'paper_id':'a'}
        a=diagnostics.feature_values(row,set());row.update(label=1,paper_id='different')
        self.assertEqual(a,diagnostics.feature_values(row,set()))
        self.assertGreater(a['conditional_markers'],0)
        self.assertGreater(a['initial_connector_fraction'],0)


if __name__=='__main__':unittest.main()
