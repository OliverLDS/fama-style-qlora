"""Recovered preflight evidence must bind every sequence and ignore no failures."""
import copy
import json
import unittest

from fama_style.evaluation import preference_token_preflight as runtime
from fama_style.research.preference_preflight_review import validate
from test_preference_token_preflight import row,encode,decode


class PreferenceRecoveryReview(unittest.TestCase):
    def fixture(self):
        rows=[];packed=[];audits=[]
        for i in range(17):
            r=row(i,str(i//2) if i<12 else str(i-6),0.5 if i<12 else 1)
            p,a=runtime.audit_pair(r,encode,decode,3,1,0);rows.append(r);packed.append(p);audits.extend(a)
        config={'packages':runtime.PACKAGES,'runtime_source_sha256':'fixed-source','max_length':1024}
        report={'complete':True,'training_called':False,'model_weights_loaded':False,'gpu_used':False,
            'actual_model_log_probabilities_computed':False,'optimizer_steps':0,'packages':runtime.PACKAGES,
            'runtime_source_sha256':'fixed-source','pairs':17,'response_sequences':34,'archived_chosen_sequences_reproduced':17,
            'max_full_tokens':max(a['full_tokens'] for a in audits),'min_supervised_tokens':min(a['supervised_tokens'] for a in audits),
            'response_padding_and_causal_scoring_contract_passed':True,'synthetic_dpo_identity_passed':True}
        state={'complete':True,'pairs_completed':17,'training_called':False,'model_weights_loaded':False,'gpu_used':False}
        return rows,config,report,state,audits,packed,1,3

    def test_complete_evidence_and_corrupted_hash_or_mask(self):
        args=self.fixture();self.assertEqual(validate(*args)['verified_response_sequences'],34)
        changed=copy.deepcopy(args);changed[4][0]['input_ids_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'hash_mask'):validate(*changed)
        changed=copy.deepcopy(args);changed[5][0]['chosen']['labels'][0]=1
        with self.assertRaises(ValueError):validate(*changed)

    def test_missing_rows_and_incompatible_execution_claims_fail(self):
        changed=self.fixture();changed[4].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):validate(*changed)
        for key in ('training_called','model_weights_loaded','gpu_used','actual_model_log_probabilities_computed'):
            changed=self.fixture();changed[2][key]=True
            with self.assertRaisesRegex(ValueError,'tokenizer_only'):validate(*changed)

    def test_archive_identity_and_shared_prompt_are_checked_separately(self):
        changed=self.fixture();changed[0][0]['archived_chosen']['input_ids_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'archived_chosen'):validate(*changed)
        changed=self.fixture();rejected=changed[5][0]['rejected'];rejected['input_ids'][1]+=1
        changed[4][1]['input_ids_sha256']=runtime.sha(json.dumps(rejected['input_ids']))
        with self.assertRaisesRegex(ValueError,'prefixes_differ'):validate(*changed)


if __name__=='__main__':unittest.main()
