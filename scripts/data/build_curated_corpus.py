"""Build a reviewed text-pair corpus; never starts training or accesses credentials.

Semantic acceptance comes exclusively from the frozen assistant decisions.
PDF screening and generator statuses alone never grant acceptance.
"""
import collections
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'archive/fama_corpus_20261019'
REVIEW=ROOT/'reviews/corpus_curation_20261019'
OUT=ROOT/'data/fama_corpus_v1'
INSTRUCTION=(
 'Rewrite the passage below in the formal academic economics style of Eugene Fama '
 'and his coauthors. Preserve every substantive claim, actor, attribution, '
 'condition, qualification, uncertainty, negation, quantity, comparison, and '
 'causal relationship. Do not add or omit claims or strengthen tentative claims. '
 'Return only one coherent paragraph, without headings, Markdown, explanations, '
 'or commentary.'
)
FORMAT='fama_multi_paper_v1'

def digest(data):return hashlib.sha256(data).hexdigest()
def load_lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def words(s):return re.findall(r'\w+',unicodedata.normalize('NFKC',s).lower())
def prompt(text):return f'{INSTRUCTION}\n\nPassage:\n{text}'
def write_lines(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def split_for(paper):return {'FF1997':'validation','FF2014':'test'}.get(paper,'train')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ARCHIVE/'manifest.json').read_text())
 for f in manifest['files']:
  assert digest((ARCHIVE/f['archive_path']).read_bytes())==f['sha256'],f['archive_path']
 archived_rows={(f['archive_path'],r['paragraph_id']):r for f in manifest['files'] if f['kind']=='dataset'
                for r in load_lines(ARCHIVE/f['archive_path'])}
 screening=load_lines(REVIEW/'screening.jsonl')
 decisions=json.loads((REVIEW/'semantic_decisions.json').read_text())
 by_id={(d['paper'],d['paragraph_id']):d for d in decisions}
 assert len(by_id)==len(decisions)
 queue={ (r['paper'],r['paragraph_id']) for r in screening if not r['screen_reasons'] }
 assert set(by_id)==queue,'Every eligible pair must have an explicit semantic decision.'
 candidates=[];ledger=[];edits=[]
 for source in screening:
  raw=source['raw'];key=(source['paper'],source['paragraph_id']);review=by_id.get(key)
  assert raw==archived_rows[(source['source_archive'],key[1])],'Screened record differs from archived original.'
  row=dict(raw);reasons=list(source['screen_reasons'])
  if review:
   assert review['reviewer']=='assistant' and review['decision'] in ('hold','accept')
   if review['decision']=='hold':reasons.append('semantic_review_hold: '+review['review'])
   for patch in review['patches']:
    field=patch['field'];before=patch['before'];after=patch['after']
    assert field in ('plain_english','original_paragraph')
    assert row[field].count(before)==1,(key,before)
    previous=row[field];row[field]=previous.replace(before,after)
    edits.append(dict(paper=key[0],paragraph_id=key[1],field=field,
      before=previous,after=row[field],reason=patch['reason'],
      before_sha256=digest(previous.encode()),after_sha256=digest(row[field].encode())))
  decision='selected' if not reasons else ('exclude' if reasons[0]=='outside_narrative_prose_task' else 'hold')
  item=dict(paper=key[0],paragraph_id=key[1],decision=decision,reasons=reasons,
    source_archive=source['source_archive'],source_sha256=source['source_sha256'],
    source_pair_sha256=digest(json.dumps([raw.get('plain_english'),raw.get('original_paragraph')],ensure_ascii=False).encode()),
    source_review_flag=raw['needs_review'],source_review_notes=raw.get('review_notes',''),
    source_text_alignment=source['source_text_alignment'],semantic_review=review,
    split=split_for(key[0]) if decision=='selected' else None)
  ledger.append(item)
  if decision=='selected':
   assert review and review['decision']=='accept'
   for field in ('plain_english','original_paragraph'):
    previous=row[field];clean=' '.join(previous.split())
    if clean!=previous:
     assert words(clean)==words(previous)
     row[field]=clean
     edits.append(dict(paper=key[0],paragraph_id=key[1],field=field,before=previous,after=clean,
      reason='Collapse layout whitespace; wording and punctuation unchanged.',kind='whitespace_normalization',
      before_sha256=digest(previous.encode()),after_sha256=digest(clean.encode())))
   candidates.append(dict(id=key,row=row,source=source,ledger=item))
 # Prioritize held-out papers so a matching training passage is discarded.
 # Compare both targets and inputs: exact matches or >=0.85 word-sequence similarity.
 # This catches lexical overlap, not every semantic paraphrase of a repeated claim.
 priority={'test':0,'validation':1,'train':2}
 candidates.sort(key=lambda x:(priority[split_for(x['id'][0])],x['id']))
 kept=[];duplicates=[]
 for candidate in candidates:
  row=candidate['row'];duplicate=None
  for prior in kept:
   for field in ('original_paragraph','plain_english'):
    a=words(row[field]);b=words(prior['row'][field])
    if min(len(a),len(b))/max(len(a),len(b))<.74:continue
    if len(set(a)&set(b))/max(1,min(len(set(a)),len(set(b))))<.65:continue
    ratio=difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
    if ratio>=.85:
     duplicate=dict(removed=list(candidate['id']),kept=list(prior['id']),field=field,similarity=ratio)
     break
   if duplicate:break
  if duplicate:
   candidate['ledger'].update(decision='hold',split=None)
   candidate['ledger']['reasons'].append('exact_or_near_duplicate_of: '+'::'.join(duplicate['kept']))
   duplicates.append(duplicate)
  else:kept.append(candidate)
 exports=[]
 for candidate in kept:
  key=candidate['id'];r=candidate['row'];s=candidate['source'];split=split_for(key[0])
  exports.append(dict(id='::'.join(key),paper_key=key[0],paper_id=r['paper_id'],paragraph_id=r['paragraph_id'],
   split=split,prompt_format=FORMAT,prompt=prompt(r['plain_english']),completion=r['original_paragraph'],
   plain_english=r['plain_english'],original_paragraph=r['original_paragraph'],
   title=r['title'],authors=r['authors'],year=r['year'],section=r.get('section'),page_location=r['page_location'],
   provenance=dict(dataset_archive=s['source_archive'],dataset_sha256=s['source_sha256'],pdf_archive=s['pdf_archive']),
   quality=dict(semantic_reviewer='assistant',source_check='normalized_pdf_text_alignment',
    full_visual_transcription_verified=False,source_review_flag=r['needs_review'],source_review_notes=r.get('review_notes',''),
    repair_count=sum(e['paper']==key[0] and e['paragraph_id']==key[1] for e in edits)),
   rhetorical_taxonomy=r.get('rhetorical_taxonomy'),reader_model_taxonomy=r.get('reader_model_taxonomy')))
 exports.sort(key=lambda r:(r['paper_key'],r['paragraph_id']))
 assert len({r['id'] for r in exports})==len(exports)
 for split in ('train','validation','test'):
  write_lines(OUT/(split+'.jsonl'),[r for r in exports if r['split']==split])
 write_lines(OUT/'curated.jsonl',exports)
 write_lines(REVIEW/'decisions.jsonl',ledger)
 write_lines(REVIEW/'repair_log.jsonl',edits)
 (REVIEW/'duplicate_report.json').write_text(json.dumps(duplicates,ensure_ascii=False,indent=2)+'\n')
 papers={r['paper_key']:r['split'] for r in exports}
 for key in papers:assert len({r['split'] for r in exports if r['paper_key']==key})==1
 split_manifest=dict(policy='Whole-paper separation; alternative source generations are never appended. Test/validation papers were available to the curator, and are not a fresh blind assessment. No model training performed.',
  paper_assignments={key:dict(split=split_for(key),selected_rows=sum(r['paper_key']==key for r in exports)) for key in sorted({r['paper'] for r in screening})},
  prompt_format=FORMAT,instruction=INSTRUCTION)
 (OUT/'split_manifest.json').write_text(json.dumps(split_manifest,ensure_ascii=False,indent=2)+'\n')
 (OUT/'rewrite_instruction.txt').write_text(INSTRUCTION+'\n')
 report=dict(version='fama_corpus_v1',date='2026-10-19',raw_records=len(screening),
  explicit_semantic_reviews=len(decisions),selected_rows=len(exports),
  split_counts=dict(collections.Counter(r['split'] for r in exports)),
  paper_counts=dict(collections.Counter(r['paper_key'] for r in exports)),
  decision_counts=dict(collections.Counter(r['decision'] for r in ledger)),
  reason_counts=dict(collections.Counter(reason.split(':')[0] for r in ledger for reason in r['reasons'])),
  field_repair_events=len(edits),repaired_selected_rows=sum(r['quality']['repair_count']>0 for r in exports),
  semantic_repair_events=sum(e.get('kind')!='whitespace_normalization' for e in edits),
  whitespace_normalization_events=sum(e.get('kind')=='whitespace_normalization' for e in edits),
  target_repair_events=sum(e['field']=='original_paragraph' for e in edits),
  semantic_decisions_sha256=digest((REVIEW/'semantic_decisions.json').read_bytes()),
  archive_manifest_sha256=digest((ARCHIVE/'manifest.json').read_bytes()),
  duplicate_holds=len(duplicates),files={name:digest((OUT/name).read_bytes()) for name in ('curated.jsonl','train.jsonl','validation.jsonl','test.jsonl','rewrite_instruction.txt','split_manifest.json')},
  review_limitations=['Assistant review is not independent certification.','PDF text alignment does not verify every glyph, punctuation mark, or original paragraph boundary.',
   'Image-only sources and uncertain mathematical text remain held.','The 1965 source is partial.','Lexical deduplication does not rule out all repeated ideas or pretraining exposure.'])
 (OUT/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('files','review_limitations','reason_counts')},indent=2))

if __name__=='__main__':main()
