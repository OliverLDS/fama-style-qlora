"""Synthetic checks for training-only, reproducible and masked discovery selection."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('discovery_builder',ROOT/'scripts/data/build_taxonomy_discovery.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def synthetic_rows():
    return [{'id':f'{paper}-{i}','paper_key':paper,'split':'train',
             'completion':'Some firms may invest if demand rises. '+'Firms report revenue. '*i,
             'plain_english':'Some firms may invest when demand rises. '+'Companies report sales. '*i}
            for paper in builder.PAPERS for i in range(1,9)]


class TaxonomyDiscoveryBatchTests(unittest.TestCase):
    def test_four_per_paper_and_length_stratum(self):
        selected=builder.select(synthetic_rows())
        self.assertEqual(len(selected),28)
        for paper in builder.PAPERS:
            group=[item for item in selected if item['row']['paper_key']==paper]
            self.assertEqual({item['length_quartile'] for item in group},{1,2,3,4})
            self.assertEqual(len({item['row']['id'] for item in group}),4)

    def test_rejects_leakage_and_duplicate_ids(self):
        rows=synthetic_rows()
        for split in ('validation','test'):
            changed=copy.deepcopy(rows);changed[0]['split']=split
            with self.assertRaises(ValueError):builder.select(changed)
        rows[1]['id']=rows[0]['id']
        with self.assertRaises(ValueError):builder.select(rows)

    def test_pack_integrity_reproducibility_and_original_preservation(self):
        rows=synthetic_rows();before=copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a',Path(folder)/'b'
            builder.build(rows,a);builder.build(rows,b)
            self.assertEqual(rows,before)
            self.assertEqual((a/'manifest.json').read_bytes(),(b/'manifest.json').read_bytes())
            manifest=json.loads((a/'manifest.json').read_text())
            for name,digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((a/name).read_bytes()).hexdigest(),digest)
            originals=[json.loads(x) for x in (a/'paragraphs_A.jsonl').read_text().splitlines()]
            references=[json.loads(x) for x in (a/'paragraphs_B.jsonl').read_text().splitlines()]
            self.assertTrue(all(set(item)=={'paragraph_id','text'} for item in originals+references))
            self.assertTrue({x['paragraph_id'] for x in originals}.isdisjoint(x['paragraph_id'] for x in references))
            key=json.loads((a/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json').read_text())
            byid={row['id']:row for row in rows};atext={x['paragraph_id']:x['text'] for x in originals};btext={x['paragraph_id']:x['text'] for x in references}
            for item in key:
                self.assertEqual(atext[item['original_item_id']],byid[item['training_id']]['completion'])
                self.assertEqual(btext[item['reference_id']],byid[item['training_id']]['plain_english'])
            for stem in ('paragraphs_A','paragraphs_B','neutral_reference_pairs'):
                full=(a/(stem+'.jsonl')).read_text().splitlines()
                chunks=[(a/f'{stem}_{i:02}.jsonl').read_text().splitlines() for i in range(1,5)]
                self.assertTrue(all(len(chunk)==7 for chunk in chunks))
                self.assertEqual(full,[line for chunk in chunks for line in chunk])
            with self.assertRaises(ValueError):builder.build(rows,a)

    def test_synthetic_output_example_has_resolvable_evidence(self):
        example=json.loads((ROOT/'docs/taxonomy_discovery_v1/synthetic_example.json').read_text())
        def visit(value):
            if isinstance(value,dict):
                if 'quote' in value:
                    self.assertGreaterEqual(example['text'].count(value['quote']),value['occurrence'])
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(example['annotation'])
        self.assertEqual(example['annotation']['provenance']['annotation_context'],'single_paragraph')


if __name__=='__main__':unittest.main()
