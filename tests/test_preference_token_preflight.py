"""Boundary/padding/causal-score and reference-policy contracts without ML."""
import copy
import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fama_style.evaluation import preference_token_preflight as m


def encode(messages,generation):
    # Synthetic tokenizer only; IDs do not represent Gemma or measured outputs.
    ids=[1]+[ord(c)+100 for c in messages[0]['content'][0]['text']]+[2]
    if len(messages)>1:ids += [ord(c)+100 for c in messages[1]['content'][0]['text']]+[3]
    return ids


def decode(ids):return ''.join(chr(i-100) for i in ids if i>=100)


def row(index=0,group='a',weight=1):
    prompt='Synthetic prompt '+str(index);chosen='Faithful response '+str(index);rejected='Erroneous response '+str(index)
    user=[{'role':'user','content':[{'type':'text','text':prompt}]}]
    prefix=encode(user,True);full=encode(user+[{'role':'assistant','content':[{'type':'text','text':chosen}]}],False)
    return {'pair_id':str(index),'example_id':group,'anchor_group_id':group,'paper_id':group,'split':'training',
        'prompt':prompt,'prompt_sha256':m.sha(prompt),'chosen':chosen,'chosen_sha256':m.sha(chosen),
        'rejected':rejected,'rejected_sha256':m.sha(rejected),'anchor_balanced_weight':weight,
        'archived_chosen':{'prompt_tokens':len(prefix),'full_tokens':len(full),'supervised_tokens':len(full)-len(prefix),
            'input_ids_sha256':m.sha(json.dumps(full)),'decoded_target_matches':True}}


class PreferenceTokenPreflight(unittest.TestCase):
    def test_transformers_structured_default_is_explicitly_overridden(self):
        calls=[]
        class StructuredDefaultTokenizer:
            def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False,return_dict=True):
                ids=encode(messages,add_generation_prompt);calls.append(return_dict)
                return {'input_ids':ids,'attention_mask':[1]*len(ids)} if return_dict else ids
        tokenizer=StructuredDefaultTokenizer()
        user=[{'role':'user','content':[{'type':'text','text':row()['prompt']}]}]
        self.assertIsInstance(tokenizer.apply_chat_template(user),dict)
        m.audit_pair(row(),lambda messages,generation:m.encode_chat(tokenizer,messages,generation),decode,3,1,0)
        self.assertEqual(calls,[True,False,False,False])
        class WrongShape:
            def apply_chat_template(self,*args,**kwargs):return {'input_ids':[1,2,3]}
        with self.assertRaisesRegex(ValueError,'flat_nonnegative'):m.encode_chat(WrongShape(),user,True)

    def test_actual_pair_masks_and_archived_exact_ids(self):
        r=row();packed,audits=m.audit_pair(r,encode,decode,3,1,0)
        self.assertEqual(len(audits),2);self.assertEqual(packed['chosen']['labels'][-1],3)
        self.assertEqual(sum(v!=-100 for v in packed['chosen']['labels']),len(r['chosen'])+1)
        bad=copy.deepcopy(r);bad['archived_chosen']['input_ids_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'not_reproduced'):m.audit_pair(bad,encode,decode,3,1,0)

    def test_template_boundary_bos_truncation_decode_and_end_marker_guards(self):
        for prefix,full,decoded,target,limit in [([1,2],[1,4,5,3],'x','x',10),
                ([1,2],[1,2,1,5,3],'x','x',10),([1,2],[1,2,5,3],'x','x',3),
                ([1,2],[1,2,5,3],'wrong','x',10),([1,2],[1,2,5,3,3],'x','x',10)]:
            with self.assertRaises(ValueError):m.mask(prefix,full,decoded,target,limit,3,1)

    def test_padding_by_position_preserves_real_end_token_even_when_pad_equals_eot(self):
        seqs=[{'input_ids':[1,2,8,3],'labels':[-100,-100,8,3]},
            {'input_ids':[1,2,8,9,3],'labels':[-100,-100,8,9,3]}]
        padded=m.left_pad(seqs,3)
        self.assertEqual(padded[0]['input_ids'][0],3);self.assertEqual(padded[0]['labels'][0],-100)
        self.assertEqual(padded[0]['labels'][-1],3);self.assertEqual(padded[0]['attention_mask'],[0,1,1,1,1])

    def test_causal_shift_excludes_prompt_and_counts_response_end_token(self):
        result=m.causal_response_score([1,2,8,3],[-100,-100,8,3],[123.,-2.,-3.])
        self.assertEqual(result,{'total_log_probability':-5.,'mean_nll':2.5,'supervised_tokens':2})
        with self.assertRaises(ValueError):m.causal_response_score([1,2],[-100,2],[0.1])
        with self.assertRaises(ValueError):m.causal_response_score([1,2],[-100,-100],[-1.])

    def test_dpo_reference_identity_sign_and_extreme_stability(self):
        self.assertAlmostEqual(m.dpo_loss(-10,-12,-10,-12),math.log(2))
        self.assertLess(m.dpo_loss(-9,-12,-10,-12),math.log(2))
        self.assertLess(m.dpo_loss(-10,-13,-10,-12),math.log(2))
        self.assertGreater(m.dpo_loss(-11,-12,-10,-12),math.log(2))
        self.assertNotAlmostEqual(m.dpo_loss(-10,-12,-10,-15),math.log(2))
        self.assertTrue(math.isfinite(m.dpo_loss(-1e6,-1,-1,-1e6)))
        with self.assertRaises(ValueError):m.dpo_loss(-1,-1,-1,-1,beta=0)

    def test_cpu_check_uses_no_tokenizer_model_or_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=[]
            for i in range(17):
                group=str(i//2) if i<12 else str(i-6)
                data.append(row(i,group,0.5 if i<12 else 1))
            path=root/'pairs.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in data))
            config={'protocol':m.PROTOCOL,'model_id':m.MODEL_ID,'model_revision':m.REVISION,'max_length':1024,
                'training_authorized':False,'packages':m.PACKAGES,'data_sha256':m.file_sha(path)}
            cfg=root/'config.json';cfg.write_text(json.dumps(config));self.assertEqual(len(m.load_inputs(path,cfg)[0]),17)
            source=Path(m.__file__)
            result=subprocess.run([sys.executable,str(source),'--data',str(path),'--config',str(cfg),'--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('no tokenizer, model, GPU or training',result.stdout)
            config['model_revision']='wrong';cfg.write_text(json.dumps(config))
            with self.assertRaises(ValueError):m.load_inputs(path,cfg)

    def test_upload_notebook_is_self_contained_and_has_no_model_or_training_calls(self):
        root=Path(__file__).resolve().parents[1];path=root/'kaggle/preference_token_preflight.ipynb'
        self.assertLess(path.stat().st_size,1_000_000);book=json.loads(path.read_text())
        code=''
        for cell in book['cells']:
            if cell['cell_type']!='code':continue
            self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
            source=''.join(cell['source'])
            if source.startswith('%pip'):continue
            if source.startswith('%%writefile'):source='\n'.join(source.splitlines()[1:])
            compile(source,'notebook','exec');code+=source+'\n'
        self.assertIn("CUDA_VISIBLE_DEVICES']=''",code)
        tree=ast.parse(Path(m.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('train','generate','backward'))
            if isinstance(node,ast.Import):self.assertNotIn('torch',[a.name for a in node.names])
        self.assertIn('from transformers import AutoTokenizer',code)


if __name__=='__main__':unittest.main()
