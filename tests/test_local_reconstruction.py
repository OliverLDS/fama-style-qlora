"""Loopback isolation, content preservation and no-generation preflight checks."""
import json
import unittest
from unittest.mock import patch
from test_neutral_reconstruction import brief
from fama_style.research import local_reconstruction as local


class LocalReconstruction(unittest.TestCase):
    def fixture(self):
        base=[]
        for i in range(3):
            rid='SYN'+str(i);value=brief();value['record_id']=rid
            for method,source in [('direct_rewrite','PRIVATE_REFERENCE_'+str(i)),
                                  ('neutral_reconstruction',json.dumps({'record_id':rid,'content_brief':value}))]:
                base.append({'record_id':rid,'method':method,'source':source})
        return base

    def test_compact_form_retains_claims_nonempty_constraints_and_edges(self):
        value=brief();text=local.compact(value)
        for claim in value['claims']:
            self.assertIn(claim['proposition'],text)
            self.assertIn(claim['claim_id'],text)
            for key,content in claim.items():
                if isinstance(content,list) and content and key!='actor_ids':
                    self.assertIn(json.dumps(content,ensure_ascii=False),text)
        for edge in value['supported_relationships']:
            self.assertIn(json.dumps(edge,sort_keys=True,ensure_ascii=False),text)
        self.assertIn('Actors: firms',text)

    def test_nine_jobs_share_contract_and_do_not_leak_reference_to_briefs(self):
        jobs=local.make_jobs(self.fixture());self.assertEqual(len(jobs),9)
        self.assertEqual(len({r['job_id'] for r in jobs}),9)
        for row in jobs:
            self.assertIn(local.INSTRUCTION,row['prompt'])
            if row['method']!='direct_rewrite':self.assertNotIn('PRIVATE_REFERENCE',row['prompt'])
        with self.assertRaises(ValueError):local.make_jobs(self.fixture()[:-1])

    def test_nonloopback_or_embedded_credentials_are_rejected_before_network(self):
        for url in ('http://example.org:8080','http://user:pass@localhost:8080','https://127.0.0.1:8080'):
            with self.assertRaises(ValueError):local.call(url,'health')

    def test_context_failure_prevents_generation(self):
        paths=[]
        def fake(base,path,payload=None):
            paths.append(path)
            return {'prompt':'rendered'} if path=='apply-template' else {'tokens':list(range(4000))}
        with patch.object(local,'call',side_effect=fake):
            with self.assertRaises(ValueError):local.preflight('http://localhost:8080',local.make_jobs(self.fixture()),4096)
        self.assertNotIn('v1/chat/completions',paths)


if __name__=='__main__':unittest.main()
