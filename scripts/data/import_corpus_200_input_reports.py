"""Join exact input Pangram reports to the completed four-condition output panel."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.detector_provider_comparison import bind
from fama_style.research.neutral_reconstruction_results import file_sha

SPEC = importlib.util.spec_from_file_location('corpus_pangram', Path(__file__).with_name('import_corpus_200_pangram.py'))
parser_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser_module)


def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def aggregate(inputs, outputs):
    if len(inputs) != 8 or len({r['example_id'] for r in inputs}) != 8:
        raise ValueError('eight_unique_inputs_required')
    by = {r['example_id']: r for r in inputs}
    summary = dict(input_texts=8, input_entirely_human=sum(r['entirely_human'] for r in inputs),
                   input_mean_human_written_share=statistics.mean(r['human_written_document_percent'] for r in inputs),
                   conditions={})
    for prompt in ['strict', 'simple']:
        for condition in ['base', 'epoch_1']:
            group = [r for r in outputs if r['prompt'] == prompt and r['condition'] == condition]
            if len(group) != 8 or {r['example_id'] for r in group} != set(by):
                raise ValueError('same_eight_outputs_required')
            summary['conditions'][prompt + '_' + condition] = dict(n=8,
                output_entirely_human=sum(r['entirely_human'] for r in group),
                human_input_stays_human=sum(by[r['example_id']]['entirely_human'] and r['entirely_human'] for r in group),
                human_input_becomes_AI_related=sum(by[r['example_id']]['entirely_human'] and not r['entirely_human'] for r in group),
                AI_related_input_becomes_human=sum(not by[r['example_id']]['entirely_human'] and r['entirely_human'] for r in group),
                AI_related_inputs_available=sum(not by[r['example_id']]['entirely_human'] for r in group),
                mean_output_human_written_share=statistics.mean(r['human_written_document_percent'] for r in group),
                mean_change_human_written_share=statistics.mean(r['human_written_document_percent'] - by[r['example_id']]['human_written_document_percent'] for r in group))
    return summary


def build(root, output):
    if output.exists(): raise ValueError('refuse_overwrite_input_report_import')
    packet = root / 'input_controls_v1'; received = root / 'input_reports_v1'
    for rel, digest in read(packet / 'manifest.json')['files'].items():
        if file_sha(packet / rel) != digest: raise ValueError('input_packet_changed')
    items = rows(packet / 'PRIVATE_KEY.jsonl')
    inputs = []; evidence = {}; seen = set()
    for entry in read(received / 'receipt.private.json'):
        digest = entry['pdf_sha256']; pdf = received / 'raw_pdfs' / (digest + '.pdf')
        extracted = received / 'extracted' / (digest + '.json')
        if file_sha(pdf) != digest or file_sha(extracted) != entry['extracted_sha256']:
            raise ValueError('report_evidence_changed')
        page = read(extracted)
        item, method = bind(page, items, digest, Path(entry['source']).name, {})
        if item['scan_id'] in seen: raise ValueError('repeat_input_report_needs_audit')
        seen.add(item['scan_id'])
        inputs.append(dict(**{k: item[k] for k in ['scan_id', 'example_id', 'text_sha256', 'paper_id', 'style_group']},
                           pdf_sha256=digest, binding_method=method, **parser_module.parse(page['text'])))
        evidence[str(pdf.relative_to(root))] = digest
        evidence[str(extracted.relative_to(root))] = entry['extracted_sha256']
    if seen != {r['scan_id'] for r in items}: raise ValueError('input_coverage_incomplete')
    previous = root / 'import_v1'
    for rel, digest in read(previous / 'manifest.json')['files'].items():
        if file_sha(previous / rel) != digest: raise ValueError('output_import_changed')
    outputs = rows(previous / 'observations.private.jsonl')
    summary = aggregate(inputs, outputs)
    summary.update(protocol='corpus_200_input_output_pangram_v1', report_count=40,
        provider_versions=sorted({r['provider_version'] for r in inputs + outputs}),
        report_dates=sorted({r['report_date'] for r in inputs + outputs if r['report_date']}),
        detector_positive_input_conversion_evaluable=any(not r['entirely_human'] for r in inputs),
        score_semantics='Displayed document authorship shares, not AI probabilities; original values unchanged',
        multiple_accounts_user_reported=True, settings_equivalence_verified=False,
        new_GPU_jobs=0, new_detector_calls=0, new_model_calls=0,
        limits=['Eight reused development cases; no estimate of general detector accuracy.',
                'All inputs already receive human-written labels, so this panel cannot measure conversion of detected AI inputs.',
                'Input retention is consistent with the output pattern; causation and author-style learning are not established.',
                'GPT-6 pairs remain usable for rewriting research; these scores do not establish data quality or invalidate all detector evaluation.'])
    paired = []
    for row in inputs:
        paired.append(dict(example_id=row['example_id'], style_group=row['style_group'], input=row,
                           outputs=[r for r in outputs if r['example_id'] == row['example_id']]))
    output.mkdir()
    (output / 'input_observations.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inputs))
    (output / 'paired.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in paired))
    write(output / 'summary.json', summary)
    write(output / 'manifest.json', dict(builder_sha256=file_sha(Path(__file__)),
        parser_sha256=file_sha(Path(parser_module.__file__)), input_manifest_sha256=file_sha(packet / 'manifest.json'),
        output_import_manifest_sha256=file_sha(previous / 'manifest.json'),
        receipt_sha256=file_sha(received / 'receipt.private.json'), evidence=evidence,
        files={p.name: file_sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.root, a.output), indent=2))
