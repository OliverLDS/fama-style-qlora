"""Real-output provenance, conservative exclusions and grouped candidate packets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import actual_error_preferences as m


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def write_rows(path,values):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(v)+'\n' for v in values))
def seal(folder):write(folder/'manifest.json',{'files':{str(p.relative_to(folder)):m.file_sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'}})


class ActualErrorPreferences(unittest.TestCase):
    def fixture(self,root):
        training=root/'training';anchors=root/'anchors';packet=root/'packet';download=root/'download';prior=root/'prior';review_path=root/'review.json'
        parents=[];cases=[];jobs=[];raw=[];previous=[];reviews=[];audit=[]
        for i in range(12):
            rid=str(i);paper=str(i%10);family=['scope','time','numbers','reasoning'][i//3]
            source=f'Synthetic source {i} contains a claim.';variant=f'A claim is present in synthetic variant {i}.'
            target=f'Synthetic human reference {i} contains the claim.';negative=f'Synthetic one-edit negative {i} lacks the claim.'
            claims=[{'claim_id':'C1','source_evidence':'claim','variant_evidence':'claim','target_evidence':'claim'}]
            ledger=m.runtime.sha(json.dumps(claims,sort_keys=True,ensure_ascii=False))
            parent={'example_id':rid,'split':'training','paper_id':paper,'author_group_id':'AG'+paper,'authors':['Synthetic author '+paper],
                'source':source,'source_sha256':m.runtime.sha(source),'target':target,'target_sha256':m.runtime.sha(target)};parents.append(parent)
            anchor=parent|{'family':family,'chosen':target,'faithful_variant':variant,'rejected':negative,'claims':claims,
                'review':{'claim_ledger_sha256':ledger,'claim_checks':[{'claim_id':'C1','preserved':True}],'added_claim_identified':False},
                'original_curation':{'original_needs_review':False,'original_needs_external_context':False,'upstream_review_notes':[]}}
            cases.append(anchor);review={'example_id':rid,'target_sha256':anchor['target_sha256'],'claim_ledger_sha256':ledger,
                'alignment':'Synthetic alignment notes.','reference_style':'Synthetic style notes.'}
            for kind,text in [('original',source),('variant',variant)]:
                job={'job_id':rid+'__'+kind,'example_id':rid,'paper_id':paper,'split':'training','family':family,'input_kind':kind,
                    'source':text,'prompt':m.runtime.prompt(text),'source_sha256':m.runtime.sha(text),'chosen':target,'rejected':negative,
                    'target_sha256':m.runtime.sha(target),'rejected_sha256':m.runtime.sha(negative)};jobs.append(job)
                output=f'Actual archived synthetic generation {i} for {kind}.'
                g={k:job[k] for k in ('job_id','example_id','paper_id','family','input_kind','source','prompt','target_sha256')}
                g.update(output=output,output_tokens=10,empty=False,output_limited=False,copied_source=False);raw.append(g)
                previous.append(g|{'stage':m.STAGE,'assessment':{'text_sha256':m.runtime.sha(output),
                    'material_claim_issue_identified':kind=='original','severity_boundary_uncertain':False}})
                review[kind]={'source_sha256':m.runtime.sha(text),'output_sha256':m.runtime.sha(output),'material_error':kind=='original',
                    'uncertain':False,'reason':'Synthetic evidence.','style_comparison':'reference_preferred_provisionally' if kind=='original' else 'not_labeled',
                    'style_notes':'Synthetic separate style note.'}
                audit.append({'job_id':g['job_id'],'candidate':'chosen','decoded_target_matches':True,'prompt_tokens':20,'full_tokens':30})
            reviews.append(review)
        parents.extend({'example_id':'unused'+str(i),'split':'training'} for i in range(34))
        write_rows(training/'training.private.jsonl',parents);(training/'development.private.jsonl').write_bytes(b'Unparseable development body, never read.')
        seal(training);write_rows(anchors/'anchors.private.jsonl',cases);seal(anchors)
        write_rows(packet/'private_key/anchors.private.jsonl',cases);job_path=packet/'upload/semantic_anchor_jobs.jsonl';write_rows(job_path,jobs)
        config={'protocol':m.runtime.PROTOCOL,'model_id':m.runtime.MODEL_ID,'model_revision':m.runtime.REVISION,
            'decoding':m.runtime.DECODING,'max_length':1024,'seed':1729,'stages':list(m.runtime.STAGES),'training_authorized':False,
            'quantization':'NF4 double quantization; FP32 compute','jobs_sha256':m.file_sha(job_path),'anchor_ids':sorted(str(i) for i in range(12)),
            'adapter_sha256':{'adapter_config.json':'fixture-config','adapter_model.safetensors':'fixture-weights'}}
        write(packet/'upload/semantic_anchor_config.json',config);seal(packet)
        report_path=download/'semantic_anchor_4b_v1/evaluation.json';write(report_path,{'complete':True,'training_called':False,'config':config,
            'runtime':{'model_id':m.runtime.MODEL_ID,'model_revision':m.runtime.REVISION},'snapshots':[{'name':m.STAGE,'complete':True,'generations':raw}]})
        write(download/'semantic_anchor_4b_v1/label_audit.json',audit)
        write_rows(prior/'per_output.private.jsonl',previous)
        write(prior/'receipt.json',{'packet_manifest_sha256':m.file_sha(packet/'manifest.json'),'download_sha256':{
            str(p.relative_to(download)):m.file_sha(p) for p in download.rglob('*') if p.is_file()}});seal(prior)
        write(review_path,{'protocol':'actual_error_preference_review_v1','independent_review':False,'semantic_gold_verified':False,'cases':reviews})
        return (training,anchors,packet,download,prior,review_path),cases,previous,reviews,parents

    def test_end_to_end_reproduces_and_never_reads_development(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,*_=self.fixture(root);read=Path.read_text
            def guarded(path,*a,**k):
                if path.name=='development.private.jsonl':raise AssertionError('Development must remain unread')
                return read(path,*a,**k)
            with patch.object(Path,'read_text',guarded):summary=m.build(*args,root/'out')
            self.assertEqual(summary['included_pairs'],12);self.assertEqual(summary['unique_included_anchors'],12)
            self.assertEqual(summary['included_papers'],10);self.assertEqual(summary['paired_sft_control_rows'],12)
            self.assertFalse(summary['training_ready']);self.assertEqual(summary['model_calls'],0)
            pairs=m.rows(root/'out/preferences.private.jsonl');sft=m.rows(root/'out/sft_control.private.jsonl')
            for p,c in zip(pairs,sft):
                self.assertEqual(p['chosen'],c['completion']);self.assertEqual(p['prompt'],c['prompt'])
                self.assertFalse(p['synthetic_rejection']);self.assertEqual(p['anchor_balanced_weight'],1)
            m.build(*args,root/'again')
            self.assertEqual({p.name:p.read_bytes() for p in (root/'out').iterdir()},
                {p.name:p.read_bytes() for p in (root/'again').iterdir()})
            with self.assertRaisesRegex(ValueError,'overwrite'):m.build(*args,root/'out')

    def test_uncertain_and_nonmaterial_outputs_cannot_become_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            _,anchors,generated,reviews,_=self.fixture(Path(temp));a=anchors[0];g=copy.deepcopy(generated[0]);r=copy.deepcopy(reviews[0])
            g['assessment']['severity_boundary_uncertain']=True;r['original']['uncertain']=True
            d=m.decision(a,g,r);self.assertEqual(d['decision'],'hold_uncertain_boundary')
            with self.assertRaises(ValueError):m.make_pair(a,g,d)
            self.assertEqual(m.decision(a,generated[1],reviews[0])['decision'],'exclude_no_clear_material_error')

    def test_changed_output_prompt_and_wrong_stage_fail_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            _,anchors,generated,reviews,_=self.fixture(Path(temp))
            for key,value in [('output','Different generation.'),('prompt','Different prompt.'),('stage','base')]:
                wrong=copy.deepcopy(generated[0]);wrong[key]=value
                with self.assertRaises(ValueError):m.decision(anchors[0],wrong,reviews[0])

    def test_training_membership_target_and_ledger_evidence_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            _,anchors,_,reviews,parents=self.fixture(Path(temp));by={r['example_id']:r for r in parents};a=anchors[0]
            m.validate_anchor(a,by,reviews[0])
            for key,value in [('split','development'),('chosen','Changed target.'),('claims',[{'claim_id':'C1','source_evidence':'absent'}])]:
                changed=copy.deepcopy(a);changed[key]=value
                with self.assertRaises(ValueError):m.validate_anchor(changed,by,reviews[0])

    def test_synthetic_negatives_and_tampered_archives_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,anchors,generated,reviews,_=self.fixture(root)
            g=copy.deepcopy(generated[0]);g['output']=anchors[0]['rejected']
            with self.assertRaisesRegex(ValueError,'synthetic_negative'):
                m.make_pair(anchors[0],g,{'decision':'include_fidelity_preference_candidate'})
            report=args[3]/'semantic_anchor_4b_v1/evaluation.json';report.write_text('{}')
            with self.assertRaisesRegex(ValueError,'archived_run_changed'):m.build(*args,root/'out')


if __name__=='__main__':unittest.main()
