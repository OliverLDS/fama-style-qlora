"""Build a private matched opening-prose corpus from uniform, page-grounded OCR.

No model call/training. Layout-derived paragraphs are candidates, not verified
transcriptions. All filters/matching use source quality/length/location, not style
scores or classification outcomes. Existing training/validation/test files stay frozen.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

from fama_style.rhetoric.concise_plan import TRAIN_SHA, file_sha, read_lines
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'author_comparison_v1'
TARGETS = {'F1965','F1980','FJ1983','FF2000','ME1997','NL2003','VL2018'}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def words(text):return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text)
def terminal(text):return bool(re.search(r'[.!?][\"\u201d\u2019\']?$',text.strip()))


def validate_registry(registry):
    papers=registry['papers'];pairs=registry['topic_pairs']
    if registry['protocol']!=PROTOCOL or len(papers)!=14 or len({p['paper_id'] for p in papers})!=14:raise ValueError('fourteen_unique_papers_required')
    targets={p['paper_id'] for p in papers if p['label']==1}
    if targets!=TARGETS or len(pairs)!=7 or len({p['topic'] for p in pairs})!=7:raise ValueError('seven_training_topics_required')
    byid={p['paper_id']:p for p in papers}
    for p in papers:
        if p['label'] not in (0,1) or not p['authors'] or not p['ocr_pages'] or p['columns'] not in (1,2):raise ValueError('paper_metadata')
        names=' '.join(p['authors']).lower()
        if p['label']==0 and any(re.search(r'\b'+re.escape(n)+r'\b',names) for n in registry['excluded_author_surnames']):raise ValueError('excluded_author_in_control')
        if p['label']==1 and 'fama' not in names:raise ValueError('target_author_missing')
    used=[]
    for pair in pairs:
        a,b=byid[pair['target_paper']],byid[pair['control_paper']]
        if a['label']!=1 or b['label']!=0 or a['topic']!=b['topic'] or a['topic']!=pair['topic']:raise ValueError('topic_pair_mismatch')
        used.extend([a['paper_id'],b['paper_id']])
    if Counter(used)!=Counter(byid.keys()):raise ValueError('papers_must_appear_once')


def normalize_header(text):return re.sub(r'[^a-z]','',text.lower())


def merge_rows(lines):
    """Join fragments of the same printed line within an already cropped column."""
    rows=[]
    for line in sorted(lines,key=lambda l:(-(l['y']+l['height']/2),l['x'])):
        center=line['y']+line['height']/2
        if rows and abs(center-rows[-1]['center'])<0.45*min(line['height'],rows[-1]['height']):
            rows[-1]['parts'].append(line)
        else:rows.append({'center':center,'height':line['height'],'parts':[line]})
    for row in rows:
        parts=sorted(row['parts'],key=lambda p:p['x']);row['text']=' '.join(p['text'].strip() for p in parts)
        row['x']=min(p['x'] for p in parts);row['width']=max(p['x']+p['width'] for p in parts)-row['x']
        row['confidence']=min(p['confidence'] for p in parts);row['line_ids']=[p['line_id'] for p in parts]
    return rows


def paragraph_candidates(paper,pages):
    repeated=Counter()
    for page in pages:
        repeated.update(set(normalize_header(l['text']) for l in page['lines'] if l['y']>.88 or l['y']<.08))
    result=[];line_drops=Counter()
    stop_references=False
    for pi,page in enumerate(sorted(pages,key=lambda p:p['pdf_page'])):
        if stop_references:break
        heights=[l['height'] for l in page['lines'] if .1<l['y']<.88 and len(words(l['text']))>=4]
        typical=statistics.median(heights) if heights else .015
        for column in range(1,paper['columns']+1):
            if stop_references:break
            lines=[]
            for l in page['lines']:
                if l.get('column',1)!=column:continue
                reason=None
                if l['y']<.06 or l['y']>.94:reason='page_margin'
                elif repeated[normalize_header(l['text'])]>=3 and (l['y']>.88 or l['y']<.1):reason='repeated_header'
                elif re.fullmatch(r'[\d\W]+',l['text']):reason='number_or_symbol_line'
                elif l['height']<.70*typical and l['y']<.32 and len(words(l['text']))>=4:reason='small_bottom_footnote'
                if reason:line_drops[reason]+=1
                else:lines.append(l)
            rows=merge_rows(lines)
            if not rows:continue
            margin=sorted(r['x'] for r in rows)[max(0,int(len(rows)*.15))]
            widths=sorted(r['width'] for r in rows);width=widths[min(len(widths)-1,int(len(widths)*.8))]
            gaps=[a['center']-b['center'] for a,b in zip(rows,rows[1:]) if .004<a['center']-b['center']<.08]
            gap=statistics.median(gaps) if gaps else typical*1.4
            block=[]
            def flush(boundary="paragraph"):
                if block:
                    raw='\n'.join(r['text'] for r in block)
                    result.append({'paper_id':paper['paper_id'],'pdf_page_start':page['pdf_page'],'pdf_page_end':page['pdf_page'],
                        'column':column,'window_position':pi/max(1,len(pages)-1),'raw_ocr':raw,
                        'line_refs':[{'pdf_page':page['pdf_page'],'column':column,'line_ids':r['line_ids']} for r in block],
                        'end_boundary':boundary,'min_ocr_confidence':min(r['confidence'] for r in block),'mean_ocr_confidence':statistics.mean(r['confidence'] for r in block)})
                    block.clear()
            for i,row in enumerate(rows):
                text=row['text']
                if re.sub(r'[^a-z]','',text.lower()) in {'references','bibliography','literaturecited'}:
                    flush('references');stop_references=True;break
                letters=[c for c in text if c.isalpha()]
                uppercase=bool(letters) and sum(c.isupper() for c in letters)/len(letters)>.8
                mathline=bool(re.search(r'[=\u2211\u222b]',text)) or len(words(text))<2
                title_words=[w for w in words(text) if w.lower() not in {'a','an','the','of','and','in','on','to','for','with'}]
                title_heading=len(title_words)>=3 and len(words(text))<14 and all(w[0].isupper() for w in title_words) and not terminal(text)
                heading=uppercase or title_heading or (len(words(text))<10 and row['width']<.67*width and not terminal(text)
                                     and i+1<len(rows) and row['center']-rows[i+1]['center']>1.25*gap)
                if heading or mathline:
                    flush('math_or_heading');line_drops['heading_or_math_boundary']+=1;continue
                if block:
                    previous=block[-1];distance=previous['center']-row['center']
                    new=(distance>1.55*gap or (terminal(previous['text']) and (row['x']>margin+.014 or previous['width']<.78*width)))
                    if new:flush()
                block.append(row)
            flush('layout_end')
    # Join only an unfinished block with a lowercase continuation, never unrelated
    # complete paragraphs. Ambiguous hyphenation is rejected later, not repaired.
    joined=[]
    for row in result:
        if joined and joined[-1]['end_boundary']=='layout_end' and not terminal(joined[-1]['raw_ocr']) and row['raw_ocr'][:1].islower() and row['pdf_page_start']-joined[-1]['pdf_page_end'] in (0,1):
            prior=joined[-1];prior['raw_ocr']+='\n'+row['raw_ocr'];prior['pdf_page_end']=row['pdf_page_end'];prior['line_refs']+=row['line_refs'];prior['end_boundary']=row['end_boundary']
            prior['min_ocr_confidence']=min(prior['min_ocr_confidence'],row['min_ocr_confidence'])
            prior['mean_ocr_confidence']=(prior['mean_ocr_confidence']+row['mean_ocr_confidence'])/2
        else:joined.append(row)
    return joined,dict(line_drops)


def supported_word(word,lexicon):
    if word in lexicon:return True
    bases=set()
    if word.endswith('s'):bases.add(word[:-1])
    if word.endswith('es'):bases.add(word[:-2])
    if word.endswith('ies'):bases.add(word[:-3]+'y')
    for suffix in ('ed','ing'):
        if word.endswith(suffix):
            stem=word[:-len(suffix)];bases.update([stem,stem+'e'])
            if len(stem)>2 and stem[-1]==stem[-2]:bases.add(stem[:-1])
    return bool(bases & lexicon)


def normalize_candidate(raw,lexicon):
    events=[];unresolved=[]
    def replacement(match):
        merged=match[1]+match[2]
        if supported_word(merged.lower(),lexicon):
            events.append({'raw_start':match.start(),'raw_end':match.end(),'before':match[0],'after':merged,
                           'method':'dictionary-supported line-break join; not independent visual verification'})
            return merged
        unresolved.append({'before':match[0],'raw_start':match.start(),'raw_end':match.end()})
        return match[0]
    text=re.sub(r'([A-Za-z]+)-[ \t]*\n([a-z]+)',replacement,raw)
    return ' '.join(unicodedata.normalize('NFKC',text).split()),events,unresolved


def quality_reason(row,policy,lexicon=None):
    raw=row['raw_ocr'];text,_,unresolved=normalize_candidate(raw,lexicon or set());n=len(words(text))
    if unresolved:return 'ambiguous_line_break_hyphen',text
    if not policy['word_range'][0]<=n<=policy['word_range'][1]:return 'word_budget',text
    if not text[:1].isupper() or not terminal(text):return 'incomplete_boundary',text
    if re.match(r'^[^.!?]{1,90}:\s+[^.!?]{1,60}\s+--',text):return 'possible_inline_heading',text
    if row['min_ocr_confidence']<.85:return 'low_line_confidence',text
    if sum(c.isalpha() for c in text)/max(1,len(text))<.72:return 'non_prose_or_table',text
    if len(re.findall(r'\d',text))/max(1,len(text))>.06:return 'numeric_density',text
    if re.match(r'^(?:fig[.]|figure\s+\d+\s*[.:-]|table\s+[ivx\d]+\s*[:.])',text,re.I):return 'figure_or_table_caption',text
    if any(ch in text for ch in '=∑∫�|'):return 'math_or_corrupt_symbol',text
    if re.search(r'\b(?:copyright|all rights reserved|jstor|downloaded from|working paper no|abstract|acknowledg|issn|http|e-mail|supra|ibid)\b',text,re.I):return 'metadata_or_abstract',text
    return None,text


def choose_matches(a,b,policy):
    """Deterministic no-replacement matching on length/location only."""
    edges=[]
    for x in a:
        for y in b:
            ratio=max(x['word_count'],y['word_count'])/min(x['word_count'],y['word_count'])
            position=abs(x['window_position']-y['window_position'])
            if ratio>policy['max_word_ratio'] or position>policy['max_window_position_difference']:continue
            cost=abs(x['word_count']-y['word_count'])+12*position
            edges.append((cost,x['id'],y['id'],x,y))
    used=set();matches=[]
    for cost,_,__,x,y in sorted(edges,key=lambda e:e[:3]):
        if x['id'] in used or y['id'] in used:continue
        used.update([x['id'],y['id']]);matches.append((x,y,cost))
        if len(matches)==policy['max_pairs_per_topic']:break
    if len(matches)<policy['min_pairs_per_topic']:raise ValueError('insufficient_matched_quality_prose')
    return matches


def build(registry_path,ocr_path,output,lexicon_path=Path("/usr/share/dict/words")):
    lexicon={w.strip().lower() for w in lexicon_path.read_text().splitlines()}
    registry=json.loads(registry_path.read_text());validate_registry(registry)
    train=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(train)!=TRAIN_SHA:raise ValueError('frozen_training_changed')
    splits=json.loads((ROOT/'data/fama_corpus_v1/split_manifest.json').read_text())['paper_assignments']
    if any(splits[p]['split']!='train' for p in TARGETS):raise ValueError('target_split_changed')
    pages=read_lines(ocr_path);by_paper=defaultdict(list)
    if len({(p['paper_id'],p['pdf_page']) for p in pages})!=len(pages):raise ValueError('duplicate_ocr_page')
    for p in pages:by_paper[p['paper_id']].append(p)
    if set(by_paper)!={p['paper_id'] for p in registry['papers']}:raise ValueError('ocr_paper_coverage')
    candidates=[];rejected=[];reports=[];pdf_hashes={}
    for paper in registry['papers']:
        path=ROOT/paper['pdf'];pdf_hashes[paper['paper_id']]=file_sha(path)
        ps=by_paper[paper['paper_id']]
        if sorted(p['pdf_page'] for p in ps)!=sorted(paper['ocr_pages']):raise ValueError('ocr_page_coverage')
        if any(p.get('columns',1)!=paper['columns'] for p in ps):raise ValueError('column_layout_not_corrected')
        rows,drops=paragraph_candidates(paper,ps);counts=Counter();accepted=[]
        for r in rows:
            reason,text=quality_reason(r,registry['paragraph_match_policy'],lexicon);r['id']='P'+sha(paper['paper_id']+'|'+json.dumps(r['line_refs'],sort_keys=True))[:20]
            if reason:counts[reason]+=1;rejected.append({**r,'reason':reason});continue
            accepted.append({**r,'text':text,'text_sha256':sha(text),'word_count':len(words(text)),
                 'label':paper['label'],'topic':paper['topic'],'authors':paper['authors'],'version_year':paper['version_year'],
                 'pdf_sha256':pdf_hashes[paper['paper_id']],'normalization':['Unicode NFKC','whitespace collapse','dictionary-supported line-break joins only'],
                 'normalization_events':normalize_candidate(r['raw_ocr'],lexicon)[1],
                 'quality_flags':{'inline_technical_notation':bool(re.search(r'[<>+=ασμ]|\b[A-Za-z]\d\b',text)),
                                  'unrecognized_lowercase_words':[w for w in words(text) if len(w)>3 and w.islower() and not supported_word(w,lexicon)]},
                 'segmentation_verified':False,'independent_transcription_verified':False,'training_ready':False})
        candidates+=accepted;reports.append({'paper_id':paper['paper_id'],'label':paper['label'],'candidate_blocks':len(rows),'eligible':len(accepted),'rejections':dict(counts),'dropped_lines':drops})
    # Remove exact duplicates globally before matching, including cross-label ones.
    frequencies=Counter(r['text_sha256'] for r in candidates);duplicate_ids={r['id'] for r in candidates if frequencies[r['text_sha256']]>1}
    pool=[r for r in candidates if r['id'] not in duplicate_ids];byid=defaultdict(list)
    for r in pool:byid[r['paper_id']].append(r)
    selected=[];pairs=[]
    for topic in registry['topic_pairs']:
        matches=choose_matches(byid[topic['target_paper']],byid[topic['control_paper']],registry['paragraph_match_policy'])
        for x,y,cost in matches:
            mid='M'+sha(PROTOCOL+'|'+x['id']+'|'+y['id'])[:20]
            pairs.append({'match_id':mid,'topic':topic['topic'],'target_id':x['id'],'control_id':y['id'],'cost':cost,
                          'word_ratio':max(x['word_count'],y['word_count'])/min(x['word_count'],y['word_count']),
                          'position_difference':abs(x['window_position']-y['window_position'])})
            selected += [{**x,'match_id':mid},{**y,'match_id':mid}]
    summary={'protocol':PROTOCOL,'status':'matched_corpus_built_not_training_data','papers':14,'topics':7,'ocr_pages':len(pages),
        'matched_pairs':len(pairs),'paragraphs':len(selected),'target_paragraphs':sum(r['label']==1 for r in selected),
        'control_paragraphs':sum(r['label']==0 for r in selected),'eligible_candidates':len(pool),'exact_duplicate_candidates_excluded':len(duplicate_ids),
        'per_paper':[r|{'selected':sum(x['paper_id']==r['paper_id'] for x in selected)} for r in reports],
        'matching_features':['word_count','relative_position_within_opening_window'],'rhetorical_scores_used_for_selection':False,
        'max_matched_word_ratio':max(p['word_ratio'] for p in pairs),'mean_word_difference':statistics.mean(abs(x['word_count']-y['word_count']) for x,y in zip(selected[::2],selected[1::2])),
        'mean_ocr_confidence_by_label':{str(label):statistics.mean(r['mean_ocr_confidence'] for r in selected if r['label']==label) for label in (0,1)},
        'segmentation_independently_verified':False,'training_ready_rows':0,'validation_rows_read':0,'reserved_test_rows_read':0,
        'model_calls':0,'gpu_jobs_started':0,'training_called':False}
    fresh(output);write_lines(output/'paragraphs.private.jsonl',selected);write_lines(output/'eligible_pool.private.jsonl',pool)
    write_lines(output/'rejected_blocks.private.jsonl',rejected);write_json(output/'matches.private.json',pairs)
    write_json(output/'registry.private.json',registry);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'lexicon_sha256':file_sha(lexicon_path),'registry_sha256':file_sha(registry_path),'ocr_sha256':file_sha(ocr_path),'pdf_sha256':pdf_hashes,'frozen_train_sha256':TRAIN_SHA,
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'matched_pairs':len(pairs),'paragraphs':len(selected),'papers':14,'quality_pool':len(pool)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--ocr-pages',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--lexicon',type=Path,default=Path('/usr/share/dict/words'))
    a=p.parse_args();build(a.registry,a.ocr_pages,a.output_dir,a.lexicon)


if __name__=='__main__':main()
