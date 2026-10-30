"""Synthetic generation coverage and lexical-diagnostic tests."""
import unittest

from fama_style.research import neutral_reconstruction_results as n


class NeutralReconstructionResults(unittest.TestCase):
    def test_generated_and_blocked_rows_have_distinct_contracts(self):
        row={'record_id':'SYN','method':'direct_rewrite','status':'blocked','text':None,'reason':'missing context'}
        n.validate_output(row,'direct_rewrite')
        row['text']='unapproved substitute'
        with self.assertRaises(ValueError):n.validate_output(row,'direct_rewrite')
        row.update(status='generated',text='Synthetic prose.',reason='')
        n.validate_output(row,'direct_rewrite')
        with self.assertRaises(ValueError):n.validate_output(row,'neutral_reconstruction')

    def test_identity_is_full_overlap_without_a_semantic_claim(self):
        text='Firms may reduce investment when costs rise. The prediction remains conditional.'
        m=n.metrics(text,text)
        self.assertEqual(m['word_sequence_overlap'],1)
        self.assertTrue(m['word_token_copy'])
        self.assertEqual(m['longest_shared_word_run'],len(n.words(text)))
        self.assertEqual(m['changed_surface_descriptor_fields'],[])

    def test_same_vocabulary_in_different_order_is_not_identical_expression(self):
        m=n.metrics('Firms buy assets. Investors sell shares.','Investors sell shares. Firms buy assets.')
        self.assertEqual(m['word_set_jaccard'],1)
        self.assertLess(m['word_sequence_overlap'],1)
        self.assertFalse(m['word_token_copy'])

    def test_repeated_numeric_mentions_are_separate_from_new_values(self):
        m=n.metrics('In 2004 firms changed. The date is known.','In 2004 firms changed. The account repeats 2004.')
        self.assertEqual(m['added_numeric_values'],[])
        self.assertEqual(m['missing_numeric_values'],[])
        self.assertEqual(m['numeric_mention_count_differences'],{'2004':1})


if __name__=='__main__':unittest.main()
