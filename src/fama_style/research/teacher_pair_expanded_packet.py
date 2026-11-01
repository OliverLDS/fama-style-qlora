"""Freeze a separate upload packet for the expanded, unexecuted teacher-pair pilot."""
import argparse
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction_results import file_sha
from fama_style.training import teacher_pair_expanded_pilot as runtime


def build(curated, output):
    if output.exists():
        raise ValueError('refuse_overwrite_expanded_packet')
    verify_files(curated)
    lineage=json.loads((curated/'lineage.json').read_text())
    output.mkdir(parents=True)
    upload=output/'upload';upload.mkdir()
    data=upload/'teacher_pair_expanded.jsonl'
    data.write_bytes((curated/'training.private.jsonl').read_bytes()+
                     (curated/'development.private.jsonl').read_bytes())
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,
        'model_revision':runtime.REVISION,'hyperparameters':runtime.HYPER,
        'decoding':runtime.DECODING,'quantization':'NF4 double quantization; FP32 compute',
        'dataset_sha256':runtime.sha(data.read_text()),
        'training_ids':[r['example_id'] for r in rows if r['split']=='training'],
        'development_ids':lineage['preserved_development_ids'],
        'training_probe_ids':lineage['preserved_training_probe_ids'],
        'gpu_execution_authorized':False}
    path=upload/'teacher_pair_expanded_config.json';runtime.write_json(path,config)
    runtime.load_inputs(data,path)
    runtime_path=Path(runtime.__file__)
    runtime.write_json(output/'plan.json',{
        'protocol':runtime.PROTOCOL,'status':'prepared_not_executed',
        'training_pairs':46,'development_pairs':8,'epochs':2,'optimizer_updates':46,
        'updates_per_epoch':23,'baseline_generations':10,'pilot_generations':31,
        'training_authorized':False,'gpu_jobs_submitted':0,
        'tokenizer_preflight':'pending; checks all 54 rows before loading weights',
        'comparison':'Practical follow-up versus v1; data volume and optimizer budget both change.',
        'causal_data_volume_effect_identified':False,'reserved_test_rows_used':0,
        'curation_manifest_sha256':file_sha(curated/'manifest.json'),
        'runtime_source_sha256':file_sha(runtime_path),
        'dataset_sha256':file_sha(data),'config_sha256':file_sha(path)})
    runtime.write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,
        'builder_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return json.loads((output/'plan.json').read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.curated,args.output)))


if __name__=='__main__':main()
