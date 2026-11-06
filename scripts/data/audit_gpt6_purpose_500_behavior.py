"""Offline copying, repetition and input-contrast audit of the completed pilot."""
import difflib
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('purpose500_review',ROOT/'scripts/data/review_gpt6_purpose_500_result.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def repetition(text):
    sentences=[re.sub(r'\s+',' ',s).strip().casefold() for s in re.split(r'(?<=[.!?])\s+',text) if len(s.split())>=6]
    return len(sentences)-len(set(sentences))


def build(download,packet):
    output=download/'behavior_audit_v1'
    if output.exists():raise ValueError('refuse_overwrite_behavior_audit')
    report,state,items,header=review.verify(download,packet)
    rows=review.rows(packet/'upload/gpt6_purpose_500_pairs.jsonl')
    from fama_style.research.neutral_reconstruction_results import words
    contrast={}
    for split,condition,label in [('training',None,'training'),('development','existing_input','development_existing'),('development','gemini_input','development_gemini')]:
        selected=[r for r in rows if r['split']==split and (condition is None or r['input_condition']==condition)]
        values=[difflib.SequenceMatcher(None,words(r['source']),words(r['target']),autojunk=False).ratio() for r in selected]
        contrast[label]=dict(pairs=len(selected),median_source_target_word_sequence_overlap=statistics.median(values))
    notes={
        4:'Conflates the true market portfolio with the tested proxy and removes the two data-coverage objections.',
        7:'Drops the archive coverage and all sample counts; adds a misleading rate-of-change characterization of air-bag inflation and repeats a sentence.',
        8:'Drops attribution detail and the historical analogy, replacing it with repeated general warnings.',
        11:'Repeats its opening sentence and drops the initial ownership/payoff comparison.',
        12:'Collapses the research-and-development/NPV discussion into a circular statement about the advice.',
        14:'Invents a contrast with prediction ability, repeats it, and omits the regulation finding.',
        19:'Adds a money-supply/purchasing-power/price causal chain absent from the supplied input.',
    }
    examples=[]
    for i,r in enumerate([x for x in items if x['snapshot']=='epoch_1'],1):
        if i in notes:
            examples.append(dict(case=i,example_id=r['example_id'],output_sha256=r['output_sha256'],
                source_sha256=review.runtime.sha(r['source']),note=notes[i],
                assessment='targeted unblinded same-agent reading; not independent or exhaustive'))
    summary=dict(protocol='gpt6_purpose_500_behavior_v1',source_target_contrast=contrast,
        snapshots={name:dict(outputs=sum(r['snapshot']==name for r in items),
            exact_word_copies=sum(r['snapshot']==name and r['source_metrics']['word_token_copy'] for r in items),
            outputs_overlap_at_least_point8=sum(r['snapshot']==name and r['source_metrics']['word_sequence_overlap']>=.8 for r in items),
            outputs_with_exact_repeated_long_sentence=sum(r['snapshot']==name and repetition(r['output'])>0 for r in items)) for name in ['base','epoch_1']},
        targeted_case_notes=len(examples),exhaustive_semantic_error_rate=None,
        interpretation='Successful execution, but copying, repetition and content loss prevent treating lower teacher-forced loss as successful style transfer.',
        hypothesis='Held-out inputs are substantially closer to their references than the new reconstruction training inputs. Test reconstruction-matched held-out inputs before attributing failure to data quality or scaling.',
        causal_hypothesis_proven=False,new_GPU_jobs=0,new_model_calls=0,new_detector_calls=0,
        limits=['Exact sentence duplication misses paraphrased repetition and repeated clauses.',
                'Near-copy development inputs are still valid robustness tests; distribution mismatch does not excuse observed failures.',
                'Targeted notes are examples, not a calibrated 48-output semantic benchmark.'])
    output.mkdir()
    review.write(output/'summary.json',summary)
    review.write_rows(output/'targeted_case_notes.private.jsonl',examples)
    review.write(output/'manifest.json',dict(builder_sha256=review.sha(Path(__file__)),
        download_manifest_sha256=review.sha(download/'download_manifest.json'),
        packet_manifest_sha256=review.sha(packet/'manifest.json'),
        files={str(p.relative_to(output)):review.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    print(json.dumps(build(ROOT/'research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1',ROOT/'data/gpt6_purpose_500_v1'),indent=2))
