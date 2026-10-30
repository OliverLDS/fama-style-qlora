"""Synthetic checks for extraction boundaries, repairs and duplicate leakage."""
import copy
import unittest

from fama_style.research import author_curation as c


def line(text, y=.7, x=.1, width=.7, height=.015, key=1):
    return {'text':text,'y':y,'x':x,'width':width,'height':height,'line_id':key,'confidence':None}


def page(number, lines):
    return {'pdf_page':number,'lines':lines,'method':'synthetic native','total_pdf_pages':2}


class AuthorCuration(unittest.TestCase):
    def test_only_supported_typographic_repairs_preserve_numbers(self):
        text,events,pending=c.repair('The mar-\nket rose 12.5 percent. A glorp-\nnex moved.',{'market'})
        self.assertIn('market rose 12.5 percent',text)
        self.assertIn('glorp- nex',text)
        self.assertEqual(len(pending),1)
        self.assertTrue(any(e['operation']=='dictionary_supported_line_break_join' for e in events))

    def test_an_appendix_reference_is_not_a_backmatter_heading(self):
        self.assertIsNone(c.STOP.fullmatch('Appendix A that various adjustments make little difference'))
        self.assertIsNone(c.STOP.fullmatch('Appendix.'))
        self.assertTrue(c.STOP.fullmatch('Appendix A'))
        self.assertTrue(c.STOP.fullmatch('References'))

    def test_adjacent_unfinished_continuation_can_join_without_guessing(self):
        blocks,drops=c.segment({'paper_id':'SYN'},[
            page(1,[line('When borrowing costs rise, firms may')]),
            page(2,[line('reduce investment. The prediction remains conditional.')])])
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0]['pdf_page_end'],2)
        self.assertIn('cross_boundary_join',blocks[0])

    def test_section_heading_blocks_cross_page_join(self):
        blocks,_=c.segment({'paper_id':'SYN'},[
            page(1,[line('When borrowing costs rise, firms may')]),
            page(2,[line('II. Evidence',y=.85,height=.03),line('reduce investment. The prediction remains conditional.')])])
        self.assertEqual(len(blocks),2)

    def test_two_line_discussion_heading_excludes_other_authors(self):
        blocks,drops=c.segment({'paper_id':'SYN'},[page(1,[
            line('Comments',y=.85,height=.03),line('and Discussion',y=.80,height=.03,key=2),
            line('Another author discusses the model. The comment offers a different interpretation.',y=.70,key=3)])])
        self.assertFalse(blocks)
        self.assertTrue(any(r['reason']=='backmatter_heading' for r in drops))

    def test_duplicate_retention_prefers_existing_training_and_logs_both(self):
        text='Firms may adjust their investment when financing costs rise. The condition does not establish an unconditional prediction.'
        rows=[{'paragraph_id':'A','paper_id':'NEW','original_paragraph':text,'existing_split':'unassigned_new_acquisition','status':'annotation_candidate','hold_reasons':[]},
              {'paragraph_id':'Z','paper_id':'OLD','original_paragraph':text,'existing_split':'training','status':'annotation_candidate','hold_reasons':[]}]
        original=copy.deepcopy(rows)
        edges=c.duplicate_groups(rows)
        self.assertEqual(len(edges),1)
        self.assertEqual(rows[0]['duplicate_of'],'Z')
        self.assertEqual(rows[1]['status'],'annotation_candidate')
        self.assertEqual([r['original_paragraph'] for r in rows],[r['original_paragraph'] for r in original])

    def test_source_line_accounting_detects_silent_loss_and_repetition(self):
        pages=[{'paper_id':'SYN','pdf_page':1,'lines':[line('Source sentence.',key=1)]}]
        record={'paper_id':'SYN','line_refs':[{'pdf_page':1,'line_id':1}]}
        self.assertEqual(c.check_line_accounting(pages,[record],[]),1)
        with self.assertRaises(ValueError):c.check_line_accounting(pages,[],[])
        with self.assertRaises(ValueError):c.check_line_accounting(pages,[record,record],[])


if __name__=='__main__':unittest.main()
