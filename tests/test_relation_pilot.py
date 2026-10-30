"""Fresh-paper/date isolation, immutable originals and staged reference exclusion."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import relation_pilot as module
from fama_style.research.human_prose_review import schema_errors
from fama_style.research.author_control_replication import verify_files


class RelationPilotTests(unittest.TestCase):
    def fixture(self):
        papers=[{'paper_id':str(p),'title':'Study '+str(p),'authors':['Author '+chr(65+p)],
            'publication_year':1950+p,'publication_year_checked':True,'publication_evidence_url':'https://example.org/source',
            'pdf_sha256':str(p),'printed_page_offset':100} for p in range(4)]
        records=[]
        for p in range(4):
            for i in range(10):
                text=('Firms use assets to produce goods and services. Investors provide funds under specified conditions. '*3)+chr(65+p)+chr(65+i)+'.'
                records.append({'paper_id':str(p),'paragraph_id':str(p)+'_'+str(i),'stratum':module.STRATA[i//5],
                    'original_paragraph':text,'original_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'pdf_sha256':str(p),'pdf_page_start':1,'pdf_page_end':1,
                    'transcription_audit':{'before':text,'after':text}})
        return {'papers':papers},records,{'papers':[]}

    def test_new_pre2020_papers_and_author_groups_enforced(self):
        registry,records,old=self.fixture();module.validate_selection(registry,records,old)
        for value in (2020,2022,None,True):
            r=copy.deepcopy(registry);r['papers'][0]['publication_year']=value
            with self.assertRaisesRegex(ValueError,'pre2020'):module.validate_selection(r,records,old)
        old={'papers':[{'title':'Another title','authors':registry['papers'][0]['authors'],'pdf_sha256':'old'}]}
        with self.assertRaisesRegex(ValueError,'author_group_overlap'):module.validate_selection(registry,records,old)
        old={'papers':[{'title':registry['papers'][0]['title'],'authors':['Someone Else'],'pdf_sha256':'old'}]}
        with self.assertRaisesRegex(ValueError,'existing_paper'):module.validate_selection(registry,records,old)

    def test_duplicate_missing_unbalanced_or_changed_original_rejected(self):
        registry,records,old=self.fixture()
        for mode in ('missing','duplicate','unbalanced','changed'):
            rows=copy.deepcopy(records)
            if mode=='missing':rows.pop()
            elif mode=='duplicate':rows[1]=rows[0]
            elif mode=='unbalanced':rows[0]['stratum']='scope_transition'
            else:rows[0]['original_paragraph']+=' Changed.'
            with self.assertRaises(ValueError):module.validate_selection(registry,rows,old)

    def test_stage2_projection_excludes_every_reference_and_paired_claim(self):
        r={'paragraph_id':'x','plain_english':'Firms use assets. Investors provide funds.',
            'original_paragraph':'PRIVATE_REFERENCE','claims':[{'proposition':'PRIVATE_REFERENCE'}],
            'original_sentences':object(),'neutral_sentences':object(),'stratum':'PRIVATE_HINT'}
        result=module.stage2_payload(r)
        self.assertEqual(set(result),{'paragraph_id','source_paragraph'})
        self.assertNotIn('PRIVATE',str(result))
        r['plain_english']=None
        with self.assertRaises(ValueError):module.stage2_payload(r)

    def test_source_only_schema_allows_both_and_abstention_but_rejects_reference(self):
        _,schema=module.output_schemas()
        row={'annotation_version':'relation_pilot_v1_stage2','paragraph_id':'x','source_paragraph':'Firms invest. Costs fall.',
            'source_sentences':[],'source_claims':[],'boundaries':[{'boundary_id':'B1','left_sentence_id':'S1',
                'right_sentence_id':'S2','relation':'association','source_evidence':[],'restrictions':[],
                'permissible':'BOTH','presentation_preference':'NONE','rationale':'Both might be reasonable.',
                'confidence':'low','merge_candidate':None,'candidate_quality':'unverified'}],
            'needs_external_context':False,'needs_review':True,'review_notes':['Candidate required before certification.'],
            'provenance':{'visible_model_label':None,'thinking_setting':None,'reference_supplied':False,'independently_verified':False}}
        self.assertEqual(schema_errors(row,schema),[])
        row['boundaries'][0]['permissible']='UNRESOLVED';self.assertEqual(schema_errors(row,schema),[])
        row['original_paragraph']='LEAK'
        self.assertTrue(schema_errors(row,schema))

    def test_package_has_eight_five_row_batches_and_no_stage2_inputs(self):
        registry,records,old=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for p in registry['papers']:
                file=root/(p['paper_id']+'.pdf');file.write_bytes(b'%PDF-SYNTHETIC')
                p['pdf']=str(file);p['pdf_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            for r in records:r['pdf_sha256']=registry['papers'][int(r['paper_id'])]['pdf_sha256']
            for name,value in (('registry.json',registry),('old.json',old)):(root/name).write_text(json.dumps(value))
            (root/'selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            module.build(root/'registry.json',root/'selection.jsonl',root/'old.json',root/'out')
            verify_files(root/'out');seen=[]
            for batch in sorted((root/'out/stage1').glob('batch_*')):
                rows=[json.loads(l) for l in (batch/'paragraphs.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),5)
                self.assertTrue(all(set(r)=={'paragraph_id','original_paragraph'} for r in rows))
                seen.extend(r['paragraph_id'] for r in rows)
            self.assertEqual(len(set(seen)),40)
            self.assertFalse((root/'out/stage2').exists())
            (root/'out/SESSION_NOTES.md').write_text('User session details.')
            verify_files(root/'out')
