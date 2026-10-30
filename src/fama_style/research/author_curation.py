"""Conservative, page-grounded source curation for the 39-paper author expansion.

CPU-only and offline. Produces human paragraph candidates for later annotation,
never synthetic inputs, semantic-equivalence labels, or training-ready pairs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata

PROTOCOL = 'author_curation_v1'
POLICY = {'word_range': [50, 350], 'minimum_ocr_line_confidence': .80,
          'near_duplicate_jaccard': .85, 'shingle_words': 5,
          'unicode': 'NFKC with explicit audit; retain raw extraction',
          'hyphens': 'join line-break word only if dictionary-supported; otherwise hold',
          'selection': 'source quality and boundaries only; no detector/style score selection',
          'split_policy': 'preserve existing roles; new works remain unassigned',
          'training_ready': False, 'semantic_verification': 'not performed'}
STOP = re.compile(r'^(?:(?:[IVX]+|\d+)[.\s]+)?(?:references(?: and notes)?|bibliography|literature cited|appendix(?:\s+[A-Z])?(?::\s*[^.!?]{1,80})?|appendices|comments and discussion)$', re.I)
INTRO = re.compile(r'^(?:(?:[IVX]+|\d+)[.\s]+)?introduction\s*$', re.I)
META = re.compile(r'\b(?:jstor|copyright|all rights reserved|issn|http|e-mail|email|working paper (?:no|series)|cowles foundation|national bureau of economic research|downloaded from|electronic copy available|jel (?:no|classification))\b', re.I)
CAPTION = re.compile(r'^(?:figure|table|chart|exhibit)\s+[A-Z\dIVX]+\b', re.I)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text):
    return re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)


def terminal(text):
    return bool(re.search(r'[.!?][\"”’\']?$', text.strip()))


def fold(text):
    return re.sub('[^a-z]', '', unicodedata.normalize('NFKD', text).casefold())


def repair(raw, lexicon):
    """Text-preserving typography only; unresolved hyphens are never guessed."""
    text = raw
    events, unresolved = [], []
    nfkc = unicodedata.normalize('NFKC', text)
    if nfkc != text:
        events.append({'operation': 'NFKC', 'before_sha256': sha(text), 'after_sha256': sha(nfkc)})
        text = nfkc
    def join(match):
        combined = match[1] + match[2]
        variants = {combined.lower(), combined.lower().rstrip('s')}
        if variants & lexicon:
            events.append({'operation': 'dictionary_supported_line_break_join', 'start': match.start(),
                           'before': match[0], 'after': combined})
            return combined
        unresolved.append({'start': match.start(), 'text': match[0]})
        return match[0]
    text = re.sub(r'([A-Za-z]+)[-\u00ad][ \t]*\n([a-z]+)', join, text)
    # Joining a soft hyphen inside a line is safe only with the same dictionary check.
    def soft(match):
        combined = match[1] + match[2]
        if combined.lower() in lexicon:
            events.append({'operation': 'dictionary_supported_soft_hyphen', 'start': match.start(),
                           'before': match[0], 'after': combined})
            return combined
        unresolved.append({'start': match.start(), 'text': match[0]})
        return match[0]
    text = re.sub(r'([A-Za-z]+)\u00ad([a-z]+)', soft, text)
    spaced = ' '.join(text.split())
    if spaced != text:
        events.append({'operation': 'whitespace_normalization', 'before_sha256': sha(text), 'after_sha256': sha(spaced)})
    return spaced, events, unresolved


def geometry_lines(page):
    """Retain explicit OCR reading regions; infer only well-supported native columns."""
    lines = [dict(line) for line in page['lines'] if line['text'].strip()]
    if lines and all('column' in line for line in lines):
        return sorted(lines, key=lambda l: (l['column'], -(l['y'] + l['height']/2), l['x']))
    left = [l for l in lines if l['x'] < .18 and .25 < l['width'] < .48]
    right = [l for l in lines if l['x'] > .47 and .25 < l['width'] < .48]
    two = len(left) >= 8 and len(right) >= 8
    for line in lines:
        line['column'] = 2 if two and line['x'] > .47 else 1
    return sorted(lines, key=lambda l: (l['column'], -(l['y'] + l['height']/2), l['x']))


def segment(paper, pages):
    """Separate headings/notes before paragraphs; never jump an excluded interval."""
    repeated = Counter()
    for page in pages:
        repeated.update({fold(l['text']) for l in page['lines'] if l['y'] > .84 or l['y'] < .08})
    blocks, removed = [], []
    section = None
    backmatter = False
    abstract_page = None
    for page in sorted(pages, key=lambda p: p['pdf_page']):
        ordered = geometry_lines(page)
        columns = sorted({l['column'] for l in ordered})
        # Each region may be a separate printed page (the rotated multitask scan).
        for column in columns:
            lines = [l for l in ordered if l['column'] == column]
            samples = [l['height'] for l in lines if .15 < l['y'] < .84 and len(words(l['text'])) >= 6]
            typical = statistics.median(samples) if samples else .015
            widths = sorted(l['width'] for l in lines if len(words(l['text'])) >= 6)
            width = widths[int((len(widths)-1)*.8)] if widths else .7
            xs = sorted(l['x'] for l in lines if len(words(l['text'])) >= 6)
            margin = xs[int((len(xs)-1)*.15)] if xs else 0
            centers = sorted([l['y']+l['height']/2 for l in lines], reverse=True)
            gaps = [a-b for a,b in zip(centers, centers[1:]) if .003 < a-b < .09]
            gap = statistics.median(gaps) if gaps else typical*1.5
            block = []
            barrier = False
            def flush(boundary):
                nonlocal barrier
                if block:
                    blocks.append({'paper_id': paper['paper_id'], 'section': section,
                                   'pdf_page_start': page['pdf_page'], 'pdf_page_end': page['pdf_page'],
                                   'column_start': column, 'column_end': column,
                                   'raw_extraction': '\n'.join(l['text'].strip() for l in block),
                                   'line_refs': [{'pdf_page': page['pdf_page'], 'column': column,
                                                  'line_id': l['line_id'], 'method': page['method']} for l in block],
                                   'ocr_confidences': [l['confidence'] for l in block if l.get('confidence') is not None],
                                   'start_barrier': barrier, 'end_boundary': boundary,
                                   'extraction_methods': [page['method']]})
                    block.clear()
                    barrier = boundary != 'paragraph'
            for position, line in enumerate(lines):
                text = line['text'].strip()
                heading = ' '.join(text.split()).strip('—– ')
                key = fold(text)
                reason = None
                if backmatter:
                    reason = 'backmatter_after_reference_appendix_or_discussion_heading'
                elif STOP.fullmatch(heading) or (fold(heading) == 'comments' and position+1<len(lines)
                                                and fold(lines[position+1]['text']) == 'anddiscussion'):
                    flush('backmatter_heading'); backmatter = True; reason = 'backmatter_heading'
                elif META.search(text):
                    reason = 'source_metadata_or_running_matter'
                elif (line['y'] > .82 or line['y'] < .10) and repeated[key] >= 3:
                    reason = 'repeated_running_header_or_footer'
                elif line['y'] < .025 or line['y'] > .975:
                    reason = 'page_margin'
                elif not words(text):
                    reason = 'standalone_number_symbol_or_rule'
                elif key == 'abstract':
                    abstract_page = page['pdf_page']; reason = 'abstract_heading'
                elif INTRO.fullmatch(heading):
                    abstract_page = None; flush('section_heading'); section = heading; reason = 'section_heading'
                elif abstract_page == page['pdf_page']:
                    reason = 'abstract_or_ambiguous_same_page_frontmatter'
                elif CAPTION.match(text):
                    reason = 'figure_or_table_caption'
                elif len(words(text)) < 12 and not terminal(text) and (
                        line['height'] > typical*1.45 and line['width'] < width*.75
                        or len([c for c in text if c.isalpha()])>3 and sum(c.isupper() for c in text)/max(1,sum(c.isalpha() for c in text))>.85
                        or re.match(r'^(?:[IVX]+[.]|\d+(?:[.]\d+)*[.)])\s+[A-Z]', heading)):
                    flush('section_heading'); section = heading; reason = 'section_heading'
                elif line['y'] < .40 and line['height'] < typical*.90:
                    reason = 'small_font_lower_page_note'
                elif line['y'] < .25 and re.match(r'^\d{1,2}[.\s]+[A-Z]', text):
                    reason = 'numbered_lower_page_note'
                if reason:
                    harmless = (reason in ('repeated_running_header_or_footer', 'page_margin') or
                                reason == 'standalone_number_symbol_or_rule' and (line['y'] < .10 or line['y'] > .90))
                    flush('layout_end' if harmless else 'excluded_line')
                    if not harmless: barrier = True
                    removed.append({'paper_id': paper['paper_id'], 'pdf_page': page['pdf_page'],
                                    'line_id': line['line_id'], 'column': column, 'text': text, 'reason': reason})
                    continue
                if block:
                    prior = block[-1]
                    distance = prior['y'] + prior['height']/2 - line['y'] - line['height']/2
                    if (distance > gap*1.48 or (terminal(prior['text']) and
                            (line['x'] > margin+.012 or prior['width'] < width*.78))):
                        flush('paragraph')
                block.append(line)
            flush('layout_end')
    joined = []
    for block in blocks:
        if joined:
            prior = joined[-1]
            adjacent = (block['pdf_page_start'] == prior['pdf_page_end'] and block['column_start'] == prior['column_end']+1
                        or block['pdf_page_start'] == prior['pdf_page_end']+1 and block['column_start'] == 1)
            if (prior['end_boundary'] == 'layout_end' and adjacent and prior['section'] == block['section']
                    and not terminal(prior['raw_extraction']) and block['raw_extraction'][:1].islower()
                    and not block['start_barrier']):
                prior['raw_extraction'] += '\n'+block['raw_extraction']
                prior['line_refs'] += block['line_refs']; prior['ocr_confidences'] += block['ocr_confidences']
                prior['pdf_page_end'] = block['pdf_page_end']; prior['column_end'] = block['column_end']
                prior['end_boundary'] = block['end_boundary']
                prior['extraction_methods'] = sorted(set(prior['extraction_methods']+block['extraction_methods']))
                prior['cross_boundary_join'] = 'unfinished-to-lowercase adjacent continuation; no intervening excluded lines'
                continue
        joined.append(block)
    return joined, removed


def screen(block, lexicon):
    text, events, unresolved = repair(block['raw_extraction'], lexicon)
    reasons, flags = [], []
    tokens = words(text)
    if unresolved: reasons.append('unresolved_line_break_or_soft_hyphen')
    if not POLICY['word_range'][0] <= len(tokens) <= POLICY['word_range'][1]: reasons.append('outside_word_scope')
    if not text[:1].isupper() or not terminal(text): reasons.append('uncertain_paragraph_boundary')
    if len(re.findall(r'[.!?](?:\s|$)', text)) < 2: reasons.append('fewer_than_two_sentence_terminators')
    if META.search(text): reasons.append('metadata_contamination')
    if CAPTION.match(text): reasons.append('caption')
    if re.match(r'^(?:notes?|sources?)\s*:',text,re.I): reasons.append('table_or_figure_note')
    if sum(c.isdigit() for c in text)/max(1,len(text)) > .08: reasons.append('table_or_numeric_density')
    if sum(c.isalpha() for c in text)/max(1,len(text)) < .65: reasons.append('low_prose_density')
    if any(ord(c)<32 or '\ue000'<=c<='\uf8ff' or c=='\ufffd' for c in text): reasons.append('unresolved_corrupt_glyph')
    if re.search(r'[=∑∫≥≤{}]|\b(?:PROPOSITION|THEOREM|LEMMA|Proof)\b', text): reasons.append('mathematical_or_proof_passage')
    confidence = min(block['ocr_confidences']) if block['ocr_confidences'] else None
    if confidence is not None and confidence < POLICY['minimum_ocr_line_confidence']: reasons.append('low_ocr_line_confidence')
    if text.startswith(('"','“','‘')): reasons.append('possible_quotation')
    if re.search(r'\b(?:table|figure|equation|appendix|section)\s*(?:\d|[IVX])|\([A-Za-z]?\d+\)', text, re.I): flags.append('external_table_equation_or_section_reference')
    if re.match(r'^(?:This|These|Such|It|They|The former|The latter)\b',text): flags.append('context_dependent_opening')
    if confidence is not None: flags.append('OCR_not_independently_verified')
    uncommon = [w for w in tokens if w.islower() and len(w)>3 and w.lower() not in lexicon]
    if len(uncommon)/max(1,len(tokens)) > .15: reasons.append('high_unrecognized_word_rate')
    return text, events, unresolved, sorted(set(reasons)), flags, confidence


def duplicate_groups(rows):
    """Exact/near duplicates across all candidate papers, before any split assignment."""
    index = defaultdict(list)
    sets = {}
    parent = {r['paragraph_id']: r['paragraph_id'] for r in rows}
    edges = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for row in sorted(rows, key=lambda r:r['paragraph_id']):
        key = row['paragraph_id']
        tokens = [w.casefold() for w in words(row['original_paragraph'])]
        shingles = {tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))}
        shared = Counter(other for shingle in shingles for other in index[shingle])
        for other, count in shared.items():
            score = count/max(1,len(shingles)+len(sets[other])-count)
            if score >= POLICY['near_duplicate_jaccard']:
                parent[find(key)] = find(other)
                edges.append({'left': key, 'right': other, 'jaccard_5gram': score})
        sets[key] = shingles
        for shingle in shingles: index[shingle].append(key)
    groups = defaultdict(list)
    for row in rows: groups[find(row['paragraph_id'])].append(row)
    for group in groups.values():
        canonical = min(group, key=lambda r:(r['existing_split']!='training', r['paragraph_id']))
        cluster = 'DC_'+sha('|'.join(sorted(r['paragraph_id'] for r in group)))[:20]
        for row in group:
            row['duplicate_cluster_id'] = cluster
            row['duplicate_of'] = canonical['paragraph_id'] if row is not canonical else None
            if row is not canonical:
                row['status'] = 'hold'; row['hold_reasons'].append('duplicate_or_near_duplicate')
    return edges


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def check_line_accounting(pages, records, removed):
    expected=Counter((p['paper_id'],p['pdf_page'],l['line_id']) for p in pages for l in p['lines'] if l['text'].strip())
    used=Counter((r['paper_id'],l['pdf_page'],l['line_id']) for r in records for l in r['line_refs'])
    used.update((r['paper_id'],r['pdf_page'],r['line_id']) for r in removed)
    if expected!=used: raise ValueError('source_lines_lost_or_counted_twice')
    return sum(expected.values())


def build(root, extraction, output, lexicon_path=Path('/usr/share/dict/words')):
    root, extraction = root.resolve(), extraction.resolve()
    if output.exists(): raise ValueError('refuse_to_overwrite_curated_snapshot')
    registry = json.loads((extraction/'source_registry.private.json').read_text())
    papers = registry['papers']
    inventory_path=root/'docs/author_inventory_v2/inventory.json'
    groups={paper_id:g['style_group'] for g in json.loads(inventory_path.read_text())['roster'] for paper_id in g['paper_ids']}
    if len(papers)!=39 or any(p['existing_split'] in ('test','reserved_test','development','validation') for p in papers):
        raise ValueError('expected_39_nonreserved_sources')
    native = [json.loads(l) for l in (extraction/'native_pages.private.jsonl').read_text().splitlines()]
    ocr = [json.loads(l) for path in sorted(extraction.glob('ocr_pages_*.private.jsonl')) for l in path.read_text().splitlines()]
    fallback = extraction/'ocr_fallback_audit_v2.private.json'
    if not fallback.exists(): fallback=extraction/'ocr_fallback_audit.private.json'
    expected = {(r['paper_id'],r['pdf_page']) for r in json.loads(fallback.read_text())}
    actual = [(r['paper_id'],r['pdf_page']) for r in ocr]
    if set(actual)!=expected or len(actual)!=len(set(actual)): raise ValueError('missing_duplicate_or_unexpected_OCR_page')
    replacements = {(r['paper_id'],r['pdf_page']):r for r in ocr}
    pages = [replacements.get((r['paper_id'],r['pdf_page']), r) for r in native]
    grouped = defaultdict(list)
    for page in pages: grouped[page['paper_id']].append(page)
    lexicon = {w.strip().lower() for w in lexicon_path.read_text().splitlines()}
    records, removed, coverage = [], [], []
    for paper in sorted(papers,key=lambda p:p['paper_id']):
        if file_sha(root/paper['pdf']) != paper['pdf_sha256']: raise ValueError('source_pdf_changed')
        sample = grouped[paper['paper_id']]
        total = sample[0]['total_pdf_pages']
        if sorted(p['pdf_page'] for p in sample)!=list(range(1,total+1)): raise ValueError('incomplete_native_page_coverage')
        coverage.append({'paper_id':paper['paper_id'],'pdf_pages':total,'pages_extracted':len(sample),
                         'ocr_pages':sum((p['paper_id'],p['pdf_page']) in replacements for p in sample),
                         'remaining_pages':[],'full_prose_verification':False})
        blocks, drops = segment(paper,sample); removed += drops
        for block in blocks:
            text, events, unresolved, reasons, flags, confidence = screen(block,lexicon)
            key = 'AC_'+sha(paper['paper_id']+'|'+json.dumps(block['line_refs'],sort_keys=True))[:20]
            records.append(block | {'paragraph_id':key,'original_paragraph':text,'original_sha256':sha(text),
                'raw_extraction_sha256':sha(block['raw_extraction']),'paper_pdf_sha256':paper['pdf_sha256'],
                'authors':paper['authors'],'title':paper['title'],'version_year':paper['version_year'],
                'style_group':groups[paper['paper_id']],
                'existing_split':paper['existing_split'],'status':'hold' if reasons else 'annotation_candidate',
                'hold_reasons':reasons,'quality_flags':flags,'needs_review':True,'training_ready':False,
                'normalization_events':events,'unresolved_hyphens':unresolved,'minimum_ocr_line_confidence':confidence,
                'native_confidence':'not measured','word_count':len(words(text)),
                'page_location':{'pdf_page_start':block['pdf_page_start'],'pdf_page_end':block['pdf_page_end'],
                                 'printed_page_start':None,'printed_page_end':None},
                'paragraph_type':'unclassified_block' if reasons else 'main_text_candidate',
                'plain_english':None,'semantic_alignment':'not_annotated'})
    candidates = [r for r in records if r['status']=='annotation_candidate']
    edges = duplicate_groups(candidates)
    accounted=check_line_accounting(pages,records,removed)
    per_paper = []
    for paper in sorted(papers,key=lambda p:p['paper_id']):
        rows = [r for r in records if r['paper_id']==paper['paper_id']]
        per_paper.append({'paper_id':paper['paper_id'],'blocks':len(rows),
                          'annotation_candidates':sum(r['status']=='annotation_candidate' for r in rows),
                          'holds':sum(r['status']=='hold' for r in rows),
                          'repair_events':sum(len(r['normalization_events']) for r in rows),
                          'hold_reasons':dict(Counter(reason for r in rows for reason in r['hold_reasons']))})
    summary = {'protocol':PROTOCOL,'papers':len(papers),'pages_extracted':len(pages),'OCR_pages':len(ocr),
               'remaining_pages':0,'paragraph_blocks':len(records),
               'annotation_candidates':sum(r['status']=='annotation_candidate' for r in records),
               'held_blocks':sum(r['status']=='hold' for r in records),'excluded_lines':len(removed),
               'normalization_events':sum(len(r['normalization_events']) for r in records),
               'source_lines_accounted_for_exactly_once':accounted,
               'cross_page_or_column_joins':sum('cross_boundary_join' in r for r in records),
               'annotation_candidates_by_style_group':dict(Counter(r['style_group'] for r in records if r['status']=='annotation_candidate')),
               'duplicate_edges':len(edges),'per_paper':per_paper,
               'removed_line_reasons':dict(Counter(r['reason'] for r in removed)),
               'training_pairs_created':0,'semantic_verification_performed':False,'training_ready':False,
               'reserved_paragraph_bodies_read':False,'model_calls':0,'GPU_jobs':0,'old_split_changes':0}
    output.mkdir(parents=True)
    write_rows(output/'all_blocks.private.jsonl',records)
    write_rows(output/'paragraphs.private.jsonl',[r for r in records if r['status']=='annotation_candidate'])
    write_rows(output/'holds.private.jsonl',[r for r in records if r['status']=='hold'])
    write_rows(output/'excluded_lines.private.jsonl',removed)
    write_rows(output/'repair_audit.private.jsonl',[{'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],
        'raw_sha256':r['raw_extraction_sha256'],'normalized_sha256':r['original_sha256'],
        'events':r['normalization_events'],'unresolved':r['unresolved_hyphens']} for r in records if r['normalization_events'] or r['unresolved_hyphens']])
    for name,value in [('summary.json',summary),('coverage.private.json',coverage),('duplicate_links.private.json',edges),
                        ('protocol.json',POLICY),('source_registry.private.json',registry)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    files = {p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
    inputs = {str(p.relative_to(root)):file_sha(p) for p in extraction.iterdir() if p.suffix in ('.json','.jsonl')}
    inputs[str(inventory_path.relative_to(root))]=file_sha(inventory_path)
    manifest = {'protocol':PROTOCOL,'files':files,'code_sha256':file_sha(Path(__file__)),
                'input_sha256':inputs,'lexicon_sha256':file_sha(lexicon_path)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--extraction',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build(args.root,args.extraction,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_paper','removed_line_reasons')},indent=2))


if __name__=='__main__':main()
