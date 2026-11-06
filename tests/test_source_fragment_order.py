from collections import Counter
from pathlib import Path
import importlib.util
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.author_curation import geometry_lines, order_ocr_fragments


class FragmentOrderTests(unittest.TestCase):
    def line(self,i,x,y,text,width=.15,column=1):
        return dict(line_id=i,x=x,y=y,width=width,height=.018,text=text,column=column)

    def test_vertical_jitter_cannot_reverse_same_baseline_fragments(self):
        left=self.line(1,.10,.700,'A given year is')
        right=self.line(2,.26,.702,'affected by costs.')
        lower=self.line(3,.10,.675,'Next printed line.',width=.6)
        lines=[right,left,lower];before=[dict(l) for l in lines]
        fixed,events=order_ocr_fragments(lines)
        self.assertEqual([l['line_id'] for l in fixed],[1,2,3])
        self.assertEqual(events[0]['before_line_ids'],[2,1])
        self.assertEqual(lines,before)
        self.assertEqual(Counter(l['text'] for l in fixed),Counter(l['text'] for l in lines))

    def test_wide_column_gap_and_separate_baselines_are_not_joined(self):
        a=self.line(1,.1,.7,'Left')
        b=self.line(2,.7,.702,'Right')
        c=self.line(3,.1,.68,'Lower')
        _,events=order_ocr_fragments([a,b,c])
        self.assertEqual(events,[])

    def test_historical_geometry_is_unchanged_without_opt_in(self):
        page={'lines':[self.line(1,.1,.7,'Left'),self.line(2,.26,.702,'Right')]}
        self.assertEqual([r['line_id'] for r in geometry_lines(page)],[2,1])
        self.assertEqual([r['line_id'] for r in geometry_lines(page,True)],[1,2])

    def test_overlapping_boxes_and_different_regions_stay_separate(self):
        a=self.line(1,.1,.7,'A',width=.4)
        b=self.line(2,.2,.702,'B',width=.4)
        self.assertEqual(order_ocr_fragments([a,b])[1],[])
        b=self.line(2,.26,.702,'B',column=2)
        self.assertEqual(order_ocr_fragments([a,b])[1],[])

    def test_recovery_patch_cannot_change_words_or_match_ambiguously(self):
        path=Path(__file__).resolve().parents[1]/'scripts/data/prepare_quality_recovery.py'
        spec=importlib.util.spec_from_file_location('quality_recovery',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.punctuation_patch('In 1987. also','1987. also','1987, also'),'In 1987, also')
        for text,before,after in [('not risky','not risky','risky'),
                                  ('bank debt','bank debt','debt bank'),
                                  ('a. a.','a.','a,')]:
            with self.assertRaises(ValueError):module.punctuation_patch(text,before,after)


if __name__=='__main__': unittest.main()
