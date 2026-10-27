"""Build a private four-row source-repair overlay; never mutate frozen corpus v1."""
import argparse
import copy
import json
from pathlib import Path
import re
from fama_style.rhetoric.concise_plan import TRAIN_SHA, file_sha, read_lines
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.rhetoric.grounded_operations import span
from fama_style.prompting.corpus import make_prompt
from .claims import sha

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'meaning_alignment_repair_v1'
PAPERS = {'NL2003', 'FJ1983', 'ME1997', 'F1980'}


def apply_edits(text, edits):
    """Resolve all occurrences against the original, then apply without overlap."""
    if not edits: raise ValueError('Empty repair')
    resolved = []
    for edit in edits:
        if not isinstance(edit['new'], str) or not edit['new'].strip() or edit['new'] == edit['old']: raise ValueError('Empty or unchanged replacement')
        bounds = span(text, edit['old'], edit['occurrence'])
        resolved.append((bounds['start'], bounds['end'], edit))
    resolved.sort(key=lambda v: v[0])
    if any(a[1] > b[0] for a, b in zip(resolved, resolved[1:])): raise ValueError('Overlapping repairs')
    chunks = []; cursor = 0; delta = 0; log = []
    for start, end, edit in resolved:
        chunks += [text[cursor:start], edit['new']]; cursor = end
        new_start = start + delta; new_end = new_start + len(edit['new']); delta += len(edit['new']) - (end-start)
        log.append({'before_span': {'start': start, 'end': end, 'quote': edit['old']},
                    'after_span': {'start': new_start, 'end': new_end, 'quote': edit['new']}, 'reason': edit['reason']})
    chunks.append(text[cursor:]); repaired = ''.join(chunks)
    for item in log:
        s = item['after_span']; assert repaired[s['start']:s['end']] == s['quote']
    # A reversible patch log, not a semantic equivalence test.
    reverse = repaired
    for item in reversed(log):
        s = item['after_span']; reverse = reverse[:s['start']] + item['before_span']['quote'] + reverse[s['end']:]
    if reverse != text: raise ValueError('Repair cannot be exactly reversed')
    return repaired, log


def repair_row(row, spec):
    if row['split'] != 'train' or row['id'] != spec['source_id'] or row['paper_key'] != spec['paper_key']: raise ValueError('Wrong parent identity/split')
    if sha(row['plain_english']) != spec['expected_source_sha256'] or sha(row['completion']) != spec['expected_target_sha256']: raise ValueError('Frozen pair changed')
    if row['completion'] != row['original_paragraph']: raise ValueError('Unexpected target contract')
    if spec['decision'] not in {'clarified', 'clarified_context_bound'}: raise ValueError('Unknown decision')
    if bool(spec['remaining_context']) != (spec['decision'] == 'clarified_context_bound'): raise ValueError('Context flag/decision mismatch')
    evidence = {}
    for edit in spec['edits']:
        if not edit['target_evidence']: raise ValueError('Repair needs target evidence')
        old = span(row['plain_english'], edit['old'], edit['occurrence'])
        evidence[(old['start'], old['end'])] = [span(row['completion'], q['quote'], q['occurrence']) for q in edit['target_evidence']]
    source, log = apply_edits(row['plain_english'], spec['edits'])
    for change in log:
        old = change['before_span']
        change['target_evidence'] = evidence[(old['start'], old['end'])]
    revised = copy.deepcopy(row)
    revised['plain_english'] = source; revised['prompt'] = make_prompt(source)
    revised['alignment_revision'] = {'protocol': PROTOCOL, 'parent_id': row['id'], 'replaces_source_sha256': sha(row['plain_english']),
      'source_sha256': sha(source), 'unchanged_target_sha256': sha(row['completion']), 'decision': spec['decision'], 'remaining_context': spec['remaining_context'],
      'source_annotations_status': 'stale_requires_reannotation', 'old_source_plan_reuse_allowed': False,
      'inherited_taxonomy_scope': 'Original target interpretation retained, not a fresh annotation of the repaired input.',
      'training_ready': False, 'automatic_generation_ready': False, 'independent_semantic_verification': False,
      'review_method': 'In-session target/context/PDF evidence review; explicit source-information repair, not an author-style augmentation.'}
    revised['quality'] = {**copy.deepcopy(row['quality']), 'meaning_alignment_patch_count': len(log),
                          'meaning_alignment_review': 'assistant_context_checked_not_independently_verified'}
    assert revised['completion'] == row['completion'] and revised['original_paragraph'] == row['original_paragraph']
    return revised, log


def normalized_pdf_anchor(text, quote):
    """Ignore case/spacing/punctuation only; no guessed OCR letter corrections."""
    def normalized(value):
        chars = []; positions = []
        for i, c in enumerate(value):
            if re.fullmatch('[a-z0-9]', c.lower()): chars.append(c.lower()); positions.append(i)
        return ''.join(chars), positions
    normalized_text, indices = normalized(text); needle, _ = normalized(quote)
    position = normalized_text.find(needle)
    if not needle or position < 0: raise ValueError('PDF text does not anchor the declared phrase')
    start = indices[position]; end = indices[position + len(needle)-1] + 1
    return {'start': start, 'end': end, 'quote': text[start:end], 'matching': 'ASCII alphanumeric normalization; not OCR correction'}


def build(spec_path, evidence_dir, output):
    train_path = ROOT/'data/fama_corpus_v1/train.jsonl'; split_path = ROOT/'data/fama_corpus_v1/split_manifest.json'
    if file_sha(train_path) != TRAIN_SHA: raise ValueError('Frozen train file changed')
    rows = {r['id']: r for r in read_lines(train_path)}; spec = json.loads(spec_path.read_text())
    if spec['protocol'] != PROTOCOL or len(spec['records']) != 4 or {r['paper_key'] for r in spec['records']} != PAPERS: raise ValueError('Expected the four flagged pairs')
    splits = json.loads(split_path.read_text())['paper_assignments']
    if any(splits[p]['split'] != 'train' for p in PAPERS): raise ValueError('Context restricted to training papers')
    archive = ROOT/'archive/fama_corpus_20261019'; archive_manifest = json.loads((archive/'manifest.json').read_text())
    archived = {f['archive_path']: f['sha256'] for f in archive_manifest['files']}
    contexts = json.loads((evidence_dir/'archive_context.private.json').read_text()); pdfs = json.loads((evidence_dir/'pdf_pages.private.json').read_text())
    if {r['paper_key'] for r in contexts} != PAPERS or {r['paper_key'] for r in pdfs} != PAPERS: raise ValueError('Unexpected context paper')
    hashes = {'train': file_sha(train_path), 'split_manifest': file_sha(split_path), 'archive_manifest': file_sha(archive/'manifest.json'), 'specification': file_sha(spec_path)}
    repaired = []; logs = []
    for s in spec['records']:
        row = rows[s['source_id']]; context = next(c for c in contexts if c['source_id'] == row['id'])
        for key in ('dataset_archive', 'pdf_archive'):
            rel = row['provenance'][key]
            if context[key] != rel or file_sha(archive/rel) != archived[rel]: raise ValueError('Archived source changed')
            hashes[rel] = file_sha(archive/rel)
        if file_sha(archive/row['provenance']['dataset_archive']) != row['provenance']['dataset_sha256']: raise ValueError('Curated provenance changed')
        archived_rows = read_lines(archive/row['provenance']['dataset_archive']); original = {r['paragraph_id']:r for r in archived_rows}
        if not any(r['paragraph_id'] == row['paragraph_id'] for r in context['records']): raise ValueError('Focal paragraph missing')
        for c in context['records']:
            if c != original[c['paragraph_id']]: raise ValueError('Context record changed')
        new, changes = repair_row(row, s)
        repaired.append(new); logs.append({'id':row['id'],'paper_key':row['paper_key'],'decision':s['decision'],'changes':changes,
           'remaining_context':s['remaining_context'],'target_changed':False,'old_source_annotations_invalidated':True})
    # Ground the four repair decisions in selected PDF text. F1980 also has a
    # targeted visual review recorded separately; no full OCR verification claimed.
    pdf_evidence = []
    for query in spec['pdf_queries']:
        paper=query['paper_key'];page=query['pdf_page'];quote=query['quote']
        if paper not in PAPERS: raise ValueError('PDF evidence must remain in training papers')
        pages = {p['pdf_page']:p['text'] for r in pdfs if r['paper_key']==paper for p in r['pages']}
        pdf_evidence.append({'paper_key':paper,'pdf_page':page,'page_text_sha256':sha(pages[page]),'anchor':normalized_pdf_anchor(pages[page],quote)})
    # Support the technical gloss from the same paper, without publishing text.
    d=spec['definition_evidence']
    if d['paper_key'] not in PAPERS: raise ValueError('Definition context must remain in training papers')
    definition = next(r for c in contexts if c['paper_key']==d['paper_key'] for r in c['records'] if r['paragraph_id']==d['paragraph_id'])
    span(definition['original_paragraph'],d['quote'],d['occurrence'])
    for p in evidence_dir.rglob('*'):
        if p.is_file(): hashes['evidence/'+str(p.relative_to(evidence_dir))] = file_sha(p)
    # All input validation precedes creation; no merged train file is emitted.
    fresh(output); write_lines(output/'repaired_pairs.private.jsonl', repaired); write_json(output/'repair_log.private.json',logs)
    write_json(output/'pdf_evidence.private.json',pdf_evidence)
    summary = {'protocol':PROTOCOL,'rows':4,'clarified':3,'clarified_context_bound':1,'source_patch_count':sum(len(l['changes']) for l in logs),
      'targets_changed':0,'frozen_train_rows_changed':0,'overlay_replacement_policy':'Replace by original ID only after explicit acceptance; never append as extra examples.',
      'training_ready_rows':0,'automatic_generation_ready_rows':0,'source_annotations_reusable':False,
      'papers':[{'paper_key':r['paper_key'],'decision':r['alignment_revision']['decision'],'patches':len(l['changes'])} for r,l in zip(repaired,logs)],
      'methods':['Archived training-paper paragraphs/context','Native macOS PDFKit text extraction; no new OCR or model','Targeted visual review of F1980 PDF pages 5-6 recorded separately'],
      'semantic_equivalence_independently_verified':False,'full_visual_transcription_verified':False,'validation_rows_read':0,'reserved_test_rows_read':0,
      'model_calls':0,'training_called':False,'next_step':'Fresh source claim/evidence bindings for the revised inputs; retain the NL2003 context flag. No reuse of earlier oracle programs without revalidation.'}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),'inputs':hashes,
      'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()},'original_train_unchanged':file_sha(train_path)==TRAIN_SHA})
    print('Created four private replacement rows: three clarified, one context-bound. Targets and corpus v1 unchanged; no generation or training.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specification',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);build(a.specification,a.evidence_dir,a.output_dir)


if __name__=='__main__': main()
