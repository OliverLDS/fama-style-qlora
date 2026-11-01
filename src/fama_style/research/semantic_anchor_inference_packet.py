"""Prepare private matched anchor jobs and frozen adapter pins; no inference."""
import argparse
import json
from pathlib import Path

from fama_style.evaluation import semantic_anchor_4b as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def build(anchors,adapter,pilot_review,output):
    if output.exists():raise ValueError('refuse_overwrite_anchor_inference_packet')
    verify_files(anchors);verify_files(pilot_review)
    cases=[json.loads(l) for l in (anchors/'anchors.private.jsonl').read_text().splitlines()]
    if len(cases)!=12 or any(r['split']!='training' for r in cases):raise ValueError('twelve_training_anchors_required')
    prior=json.loads((pilot_review/'summary.json').read_text())
    receipt=json.loads((pilot_review/'receipt.json').read_text())
    if (prior['execution_complete'] is not True or prior['optimizer_steps']!=46
            or prior['checkpoints']['2']['sha256']!=file_sha(adapter/'adapter_model.safetensors')):
        raise ValueError('reviewed_expanded_epoch_2_required')
    for name in ('adapter_config.json','adapter_model.safetensors'):
        if receipt['download_files_sha256']['teacher_pair_pilot_v2/epoch_2/adapter/'+name]!=file_sha(adapter/name):
            raise ValueError('reviewed_adapter_file_changed')
    jobs=[]
    for i,row in enumerate(cases):
        for kind in (('original','variant') if i%2 else ('variant','original')):
            source=row['source'] if kind=='original' else row['faithful_variant']
            jobs.append({'job_id':row['example_id']+'__'+kind,'example_id':row['example_id'],'paper_id':row['paper_id'],
                'split':'training','family':row['family'],'input_kind':kind,'source':source,'prompt':runtime.prompt(source),
                'source_sha256':runtime.sha(source),'chosen':row['chosen'],'rejected':row['rejected'],
                'target_sha256':row['target_sha256'],'rejected_sha256':row['rejected_sha256']})
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        'decoding':runtime.DECODING,'max_length':runtime.MAX_LENGTH,'seed':1729,'stages':list(runtime.STAGES),
        'quantization':'NF4 double quantization; FP32 compute','training_authorized':False,
        'anchor_ids':sorted(r['example_id'] for r in cases),
        'adapter_sha256':{name:file_sha(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}}
    runtime.verify_adapter(adapter,config)
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir();(output/'private_key').mkdir()
    jobs_path=upload/'semantic_anchor_jobs.jsonl';config_path=upload/'semantic_anchor_config.json'
    write_rows(jobs_path,jobs);config['jobs_sha256']=file_sha(jobs_path);write_json(config_path,config)
    runtime.load_inputs(jobs_path,config_path)
    (output/'private_key/anchors.private.jsonl').write_bytes((anchors/'anchors.private.jsonl').read_bytes())
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','anchors':12,'training_papers':len({r['paper_id'] for r in cases}),
        'input_conditions':['original','variant'],'stages':list(runtime.STAGES),'generations_planned':48,
        'completion_scores_planned':96,'pairwise_likelihood_decisions_planned':48,'tokenizer_audit_sequences_planned':48,
        'training_called':False,'optimizer_steps':0,'training_authorized':False,'gpu_jobs_submitted':0,
        'runtime_source_sha256':file_sha(Path(runtime.__file__)),'jobs_sha256':file_sha(jobs_path),'config_sha256':file_sha(config_path),
        'anchor_manifest_sha256':file_sha(anchors/'manifest.json'),'pilot_review_manifest_sha256':file_sha(pilot_review/'manifest.json'),
        'adapter_sha256':config['adapter_sha256'],'adapter_slug':'oliverlds/teacher-pair-expanded-pilot-4b','adapter_version':1,
        'development_records_used':0,'reserved_records_used':0,
        'limits':['Training-side robustness and likelihood diagnostic; adapter has seen all human targets and original inputs.',
            'New variants and synthetic negatives have same-reviewer provisional semantic validation.',
            'Conditional likelihood preference is not a calibrated truth judgment; mean/total scores have length biases.',
            'Generation prompts contain source text and ordinary rewrite instruction only; no target, rejection or oracle ledger.']}
    write_json(output/'plan.json',plan)
    write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,'builder_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('anchors','adapter','pilot-review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.anchors,args.adapter,args.pilot_review,args.output)))


if __name__=='__main__':main()
