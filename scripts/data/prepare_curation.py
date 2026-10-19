"""Screen archived sources for a conservative first prose corpus, without ML imports.

Screening only prioritizes rows; a separate explicit semantic decision is required.
Run with the temporary PDF-review Python environment (pypdf).
"""
import collections
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'archive/fama_corpus_20261019'
OUT=ROOT/'reviews/corpus_curation_20261019'

def normalize(s):
 return ''.join(re.findall('[a-z0-9]',unicodedata.normalize('NFKC',s).lower()))

def numbers(s):
 return collections.Counter(re.findall(r'(?<![A-Za-z])[-−]?\d+(?:\.\d+)?',s.replace(',','')))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ARCHIVE/'manifest.json').read_text())['files'];sources={}
 for f in manifest:sources.setdefault(f['paper'],{})[f['kind']]=f
 all_rows=[];queue=[]
 for key,files in sources.items():
  pages=[p.extract_text() or '' for p in PdfReader(ARCHIVE/files['pdf']['archive_path']).pages]
  counts=collections.Counter()
  for line in (ARCHIVE/files['dataset']['archive_path']).read_text().splitlines():
   r=json.loads(line);a=r.get('original_paragraph');b=r.get('plain_english');reasons=[]
   if r['paragraph_type']!='main_text':reasons.append('outside_narrative_prose_task')
   elif r['rewrite_status']=='already_clear':reasons.append('reserve_already_clear')
   elif r['rewrite_status']!='simplified' or not b:reasons.append('pending_rewrite')
   else:
    if not 30<=len(a.split())<=250:reasons.append('outside_v1_30_to_250_word_scope')
    if re.search(r'[=<>^{}\u0370-\u03ff\ue000-\uf8ff\u0700-\u074f\ufffd]',a+b):reasons.append('notation_requires_visual_verification')
    if re.match(r'(?i)\s*(where\b|and\b|or\b|that is\b|\d+[.)])',a) or a.rstrip().endswith((':',' is',' are')):reasons.append('context_dependent_fragment')
    if numbers(a)!=numbers(b):reasons.append('numeric_or_citation_alignment_requires_review')
    # Optional taxonomy availability is not a text-pair quality defect.
    note=r.get('review_notes','').strip()
    taxonomy_only=(not note or (('taxonomy' in note.lower() or 'taxonomies' in note.lower()) and not re.search(r'(?i)OCR|cross.page|cross.column|drop.cap|numeric|indices|glyph|superscript|hyphen|paragraph split|restored|boundary',note)))
    if r.get('needs_review') and not taxonomy_only:reasons.append('unresolved_generator_text_review')
   loc=r['page_location'];start,end=loc['pdf_page_start'],loc['pdf_page_end']
   coverage=None
   if not reasons:
    if all(not p.strip() for p in pages[start-1:end]):reasons.append('image_only_source_not_yet_proofread')
    else:
     target=normalize(a);text=normalize(' '.join(pages[start-1:end]))
     coverage=sum(m.size for m in difflib.SequenceMatcher(None,target,text,autojunk=False).get_matching_blocks())/len(target)
     if coverage<.995:reasons.append('source_text_alignment_requires_visual_review')
   obj=dict(paper=key,paragraph_id=r['paragraph_id'],source_archive=files['dataset']['archive_path'],source_sha256=files['dataset']['sha256'],
            pdf_archive=files['pdf']['archive_path'],source_text_alignment=coverage,screen_reasons=reasons,raw=r)
   all_rows.append(obj)
   if not reasons:queue.append(obj);counts['semantic_queue']+=1
   else:counts['held_or_excluded']+=1
  print(key,dict(counts),flush=True)
 for name,rows in [('screening',all_rows),('semantic_queue',queue)]:
  (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 print('Total',len(all_rows),'semantic queue',len(queue))

if __name__=='__main__':main()
