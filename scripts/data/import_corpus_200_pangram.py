"""Import frozen strict/simple prompt Pangram PDFs locally, without detector calls."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.detector_provider_comparison import bind
from fama_style.research.neutral_reconstruction_results import file_sha, metrics

SUMMARIES = {
    'We believe that this entire text is human-written.': 'Human Written',
    'We believe that this entire text is AI.': 'AI Generated',
    'We believe that this entire text is AI-assisted.': 'AI Assisted',
    'We believe that this text is a mix of AI and AI-assisted content.': 'Mix: AI / AI-assisted',
    'We believe that this text is a mix of AI and human-written content.': 'Mix: AI / human',
    'We believe that this text is a mix of AI-assisted and human-written content.': 'Mix: AI-assisted / human',
    'We believe this text is mainly AI, with some AI-assisted content.': 'Mainly AI; some AI-assisted',
    'We believe this text is mainly AI-assisted, with some AI content.': 'Mainly AI-assisted; some AI',
    'We believe this text is mainly AI-assisted, with some human-written content.': 'Mainly AI-assisted; some human',
}


def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
def lines(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse(text):
    """Keep native mixed summaries and displayed shares; do not renormalize."""
    summaries = re.findall(r'We believe[^\n]+', text)
    values = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*of the document', text)
    versions = set(re.findall(r'Pangram (\d+(?:\.\d+)+)', text))
    dates = re.findall(r'\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b', text)
    if len(summaries) != 1 or summaries[0] not in SUMMARIES or len(values) != 3 or len(versions) != 1:
        raise ValueError('unrecognized_header_preserve_without_guessing')
    values = list(map(float, values))
    if any(not 0 <= n <= 100 for n in values): raise ValueError('invalid_share')
    count = re.search(r'\b(\d+) Words\s+Pangram', text)
    return dict(provider_summary=summaries[0], provider_label=SUMMARIES[summaries[0]],
                entirely_human=summaries[0] == 'We believe that this entire text is human-written.',
                ai_generated_document_percent=values[0], ai_assisted_document_percent=values[1],
                human_written_document_percent=values[2], displayed_percentage_sum=sum(values),
                percentages_rescaled=False, provider_version=next(iter(versions)),
                report_date=datetime.strptime(dates[0], '%b %d, %Y').date().isoformat() if dates else None,
                report_timezone=None, provider_word_count=int(count[1]) if count else None,
                score_semantics='Displayed authorship shares, not document AI probability')


def build(received, strict, simple, output):
    if output.exists(): raise ValueError('refuse_overwrite_import')
    packets = {'strict': strict, 'simple': simple}; items = {}; reports = []
    for name, packet in packets.items():
        for rel, digest in read(packet / 'manifest.json')['files'].items():
            if file_sha(packet / rel) != digest: raise ValueError('frozen_packet_changed')
        items[name] = lines(packet / 'PRIVATE_KEY.jsonl')
        if len(items[name]) != 16: raise ValueError('sixteen_frozen_outputs_required')
    ids = {name: {(r['example_id'], r['condition']) for r in rows} for name, rows in items.items()}
    if ids['strict'] != ids['simple'] or len(ids['strict']) != 16: raise ValueError('same_eight_matched_cases_required')
    reviews = {
        'strict': lines(strict.parent / 'review_v1/outputs_and_assessments.private.jsonl'),
        'simple': lines(simple.parent / 'outputs_and_assessments.private.jsonl'),
    }
    receipt = read(received / 'receipt.private.json')
    observed = set(); evidence = {}
    for entry in receipt:
        prompt = entry['prompt']; digest = entry['pdf_sha256']
        pdf = received / 'raw_pdfs' / (digest + '.pdf')
        extracted = received / 'extracted' / (digest + '.json')
        if file_sha(pdf) != digest or file_sha(extracted) != entry['extracted_sha256']:
            raise ValueError('received_evidence_changed')
        page = read(extracted)
        item, method = bind(page, items[prompt], digest, Path(entry['source']).name, {})
        if file_sha(packets[prompt] / 'texts' / (item['scan_id'] + '.txt')) != item['text_sha256']:
            raise ValueError('scan_text_changed')
        key = (prompt, item['scan_id'])
        if key in observed: raise ValueError('repeat_report_needs_separate_audit')
        observed.add(key)
        original = next(r for r in reviews[prompt] if r['example_id'] == item['example_id'] and r['snapshot'] == item['condition'])
        if original['output_sha256'] != item['text_sha256']: raise ValueError('review_text_changed')
        source_metrics = metrics(original['source'], item['text'])
        reports.append(dict(prompt=prompt, **{k: item[k] for k in
                       ['scan_id', 'example_id', 'paper_id', 'style_group', 'condition', 'text_sha256']},
                       pdf_sha256=digest, binding_method=method, assessment=item['assessment'],
                       source_word_sequence_overlap=source_metrics['word_sequence_overlap'],
                       **parse(page['text'])))
        evidence[str(pdf.relative_to(received))] = digest
        evidence[str(extracted.relative_to(received))] = entry['extracted_sha256']
    if observed != {(name, r['scan_id']) for name, group in items.items() for r in group}:
        raise ValueError('incomplete_report_coverage')
    summary = dict(protocol='corpus_200_prompt_pangram_v1', reports=len(reports), cases=8,
                   conditions={}, provider_versions=sorted({r['provider_version'] for r in reports}),
                   report_dates=sorted({r['report_date'] for r in reports if r['report_date']}),
                   binding_methods=dict(Counter(r['binding_method'] for r in reports)),
                   detector_calls_by_agent=0, training_called=False, scores_are_AI_probabilities=False,
                   account_provenance={'multiple_accounts_user_reported': True,
                       'account_count': None, 'per_report_account_mapping': None,
                       'settings_equivalence_independently_verified': False,
                       'account_identities_collected': False},
                   limits=['Eight reused development cases, not a population estimate or reserved test.',
                           'Human-written labels do not establish distinctive author style or factual correctness.',
                           'New input-only controls were not scanned; detector benefit from copying cannot be isolated.',
                           'Displayed shares can sum above 100 because of small display floors; values retained unchanged.',
                           'Semantic flags are provisional automated judgments, not a detector eligibility gate.'])
    for prompt in packets:
        for condition in ['base', 'epoch_1']:
            group = [r for r in reports if r['prompt'] == prompt and r['condition'] == condition]
            if len(group) != 8: raise ValueError('condition_coverage')
            summary['conditions'][prompt + '_' + condition] = dict(n=8,
                entirely_human=sum(r['entirely_human'] for r in group),
                native_labels=dict(Counter(r['provider_label'] for r in group)),
                mean_human_written_share=statistics.mean(r['human_written_document_percent'] for r in group),
                median_human_written_share=statistics.median(r['human_written_document_percent'] for r in group),
                mean_AI_generated_share=statistics.mean(r['ai_generated_document_percent'] for r in group),
                mean_AI_assisted_share=statistics.mean(r['ai_assisted_document_percent'] for r in group),
                median_source_word_sequence_overlap=statistics.median(r['source_word_sequence_overlap'] for r in group),
                outputs_with_material_semantic_flags=sum(r['assessment']['material_claim_change'] for r in group),
                entirely_human_with_material_semantic_flags=sum(r['entirely_human'] and r['assessment']['material_claim_change'] for r in group))
    output.mkdir()
    (output / 'observations.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reports))
    write(output / 'summary.json', summary)
    write(output / 'manifest.json', dict(builder_sha256=file_sha(Path(__file__)),
          parser_binding_sha256=file_sha(ROOT / 'src/fama_style/research/detector_provider_comparison.py'),
          input_packet_manifests={name: file_sha(packet / 'manifest.json') for name, packet in packets.items()},
          receipt_sha256=file_sha(received / 'receipt.private.json'), evidence=evidence,
          files={p.name: file_sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['received', 'strict', 'simple', 'output']: parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(build(a.received, a.strict, a.simple, a.output), indent=2))
