"""CPU guards for inference score reconstruction and all-condition masked review."""
import copy
import unittest

from fama_style.evaluation.semantic_anchor_4b import sha,margin
from fama_style.research.semantic_anchor_inference_review import validate_execution,join_assessments,compare


class AnchorInferenceReview(unittest.TestCase):
    def fixture(self):
        jobs=[];anchors=[];bundle=[];key=[];assess=[];prefs=[]
        for i in range(12):
            rid=str(i);family=('scope','time','numbers','reasoning')[i//3]
            original=f'Original input {i}.';variant=f'Changed wording {i}.';target=f'Faithful target {i}.'
            claims=[{'claim_id':'C1','statement':'Synthetic statement.','constraints':{}}]
            anchors.append({'example_id':rid,'source':original,'faithful_variant':variant,'chosen':target,'claims':claims})
            bundle.append({'example_id':rid,'original_input':original,'variant_input':variant,'reference':target,'claims':claims,'outputs':{}})
            prefs.append({'example_id':rid,'ranking_best_first':[['A'],['B'],['C'],['D']]})
            for kind,source in [('original',original),('variant',variant)]:
                jobs.append({'job_id':rid+'__'+kind,'example_id':rid,'paper_id':'paper'+rid,'family':family,
                    'input_kind':kind,'source':source,'prompt':'Rewrite '+source,'chosen':target,
                    'rejected':f'Wrong target {i}.','target_sha256':sha(target)})
        audit=[{'job_id':j['job_id'],'candidate':c,'decoded_target_matches':True,'prompt_tokens':40,
            'full_tokens':100,'supervised_tokens':60,'input_ids_sha256':'a'*64} for j in jobs for c in ('chosen','rejected')]
        snaps=[]
        for stage in ('base','expanded_epoch_2'):
            snap={'name':stage,'complete':True,'generations':[],'scores':[],'contrasts':[]}
            for job in jobs:
                label={('base','original'):'A',('base','variant'):'B',('expanded_epoch_2','original'):'C',('expanded_epoch_2','variant'):'D'}[(stage,job['input_kind'])]
                text=stage+' '+job['job_id']+' output.'
                g={k:job[k] for k in ('job_id','example_id','paper_id','family','input_kind','source','prompt','target_sha256')}
                g.update({'output':text,'output_tokens':20,'empty':False,'copied_source':False,'output_limited':False});snap['generations'].append(g)
                scored=[]
                for candidate,loss in [('chosen',2.),('rejected',3.)]:
                    score={'job_id':job['job_id'],'candidate':candidate,'completion_sha256':sha(job[candidate]),'mean_nll':loss,'supervised_tokens':60,'total_nll':60*loss}
                    snap['scores'].append(score);scored.append(score)
                snap['contrasts'].append({k:job[k] for k in ('job_id','example_id','family','input_kind')}|margin(*scored))
                rid=job['example_id'];bundle[int(rid)]['outputs'][label]=text
                key.append({'example_id':rid,'label':label,'stage':stage,'job_id':job['job_id'],'input_kind':job['input_kind']})
                assess.append({'example_id':rid,'label':label,'text_sha256':sha(text),'material_claim_issue_identified':label!='D',
                    'severity_boundary_uncertain':False,'notes':'Synthetic review.'})
            snaps.append(snap)
        config={};report={'complete':True,'config':config,'training_called':False,'snapshots':snaps}
        state={'complete':True,'training_called':False,'optimizer_steps':0,'completed_generations':48,'completed_completion_scores':96}
        return report,state,audit,jobs,config,anchors,bundle,key,assess,prefs

    def test_complete_scoring_and_recomputed_signs_required(self):
        report,state,audit,jobs,config,*_=self.fixture();validate_execution(report,state,audit,jobs,config)
        for kind in ('missing','score_hash','margin','tokens'):
            r=copy.deepcopy(report)
            if kind=='missing':r['snapshots'][0]['scores'].pop()
            elif kind=='score_hash':r['snapshots'][0]['scores'][0]['completion_sha256']='wrong'
            elif kind=='tokens':r['snapshots'][0]['scores'][0]['supervised_tokens']=61
            else:r['snapshots'][0]['contrasts'][0]['mean_nll_preference']='rejected'
            with self.assertRaises(ValueError):validate_execution(r,state,audit,jobs,config)
        with self.assertRaises(ValueError):validate_execution(report,state,audit[:-1],jobs,config)
        bad=copy.deepcopy(state);bad['optimizer_steps']=1
        with self.assertRaises(ValueError):validate_execution(report,bad,audit,jobs,config)

    def test_masked_targets_claims_text_hashes_and_all_four_labels_are_bound(self):
        r,s,a,j,c,anchors,bundle,key,assess,prefs=self.fixture()
        output=join_assessments(r,j,anchors,bundle,key,assess,prefs);self.assertEqual(len(output),48)
        for kind in ('target','claim','output'):
            bad=copy.deepcopy(bundle)
            if kind=='target':bad[0]['reference']='wrong'
            elif kind=='claim':bad[0]['claims'][0]['statement']='wrong'
            else:bad[0]['outputs']['A']='wrong'
            with self.assertRaises(ValueError):join_assessments(r,j,anchors,bad,key,assess,prefs)
        with self.assertRaises(ValueError):join_assessments(r,j,anchors,bundle,key[:-1],assess,prefs)
        bad=copy.deepcopy(prefs);bad[0]['ranking_best_first']=[['A','A','B','C']]
        with self.assertRaises(ValueError):join_assessments(r,j,anchors,bundle,key,assess,bad)

    def test_paired_transitions_keep_same_cases(self):
        r,s,a,j,c,anchors,bundle,key,assess,prefs=self.fixture()
        output=join_assessments(r,j,anchors,bundle,key,assess,prefs)
        old=[g for g in output if g['stage']=='expanded_epoch_2' and g['input_kind']=='original']
        new=[g for g in output if g['stage']=='expanded_epoch_2' and g['input_kind']=='variant']
        result=compare(old,new);self.assertEqual(result['material_issue_transitions'],{'improved':12})
        with self.assertRaises(ValueError):compare(old,new[:-1])


if __name__=='__main__':unittest.main()
