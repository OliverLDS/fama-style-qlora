"""Twelve one-for-one training-input replacements with fixed targets and budget."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from fama_style.training import teacher_pair_input_replacement as runtime
from fama_style.training import teacher_pair_expanded_pilot as control_runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def replace_inputs(records,anchors):
    if (len(records)!=54 or len({r['example_id'] for r in records})!=54
            or Counter(r['split'] for r in records)!=Counter({'training':46,'development':8})):
        raise ValueError('frozen_46_training_8_development_required')
    by={r['example_id']:r for r in records};selected={a['example_id']:a for a in anchors}
    if (len(anchors)!=12 or len(selected)!=12 or not set(selected)<=set(by)
            or Counter(a['family'] for a in anchors)!=Counter({'scope':3,'time':3,'numbers':3,'reasoning':3})):
        raise ValueError('same_twelve_balanced_anchor_replacements_required')
    output=[];audit=[]
    for row in records:
        changed=copy.deepcopy(row)
        if row['example_id'] in selected:
            anchor=selected[row['example_id']]
            if (row['split']!='training' or anchor['split']!='training'
                    or row['source']!=anchor['source'] or row['target']!=anchor['chosen']
                    or row['source_sha256']!=anchor['source_sha256'] or row['target_sha256']!=anchor['target_sha256']
                    or runtime.sha(anchor['faithful_variant'])!=anchor['variant_sha256']):
                raise ValueError('replacement_anchor_not_bound_to_original_training_pair')
            changed['source']=anchor['faithful_variant'];changed['source_sha256']=anchor['variant_sha256']
            changed['prompt']=runtime.prompt(changed['source'])
            audit.append({'example_id':row['example_id'],'paper_id':row['paper_id'],'family':anchor['family'],
                'original_source':row['source'],'variant_source':changed['source'],'target':row['target'],
                'original_source_sha256':row['source_sha256'],'replacement_source_sha256':changed['source_sha256'],
                'unchanged_target_sha256':row['target_sha256'],'original_anchor_review':anchor['review'],
                'original_curation':anchor['original_curation'],'semantic_gold_verified':False})
        if any(changed[k]!=row[k] for k in row if k not in ('source','source_sha256','prompt')):
            raise ValueError('only_three_source_fields_may_change')
        output.append(changed)
    if sum(a['source']!=b['source'] for a,b in zip(records,output))!=12:raise ValueError('exactly_twelve_source_changes_required')
    return output,audit


def build(control,anchors,baseline,baseline_review,pilot_review,output):
    if output.exists():raise ValueError('refuse_overwrite_input_replacement_packet')
    for folder in (control,anchors,baseline_review,pilot_review):verify_files(folder)
    old_path=control/'upload/teacher_pair_expanded.jsonl';old_config_path=control/'upload/teacher_pair_expanded_config.json'
    groups,control_config=control_runtime.load_inputs(old_path,old_config_path)
    control_runtime.validate_baseline(baseline,control_config)
    baseline_receipt=json.loads((baseline_review/'receipt.json').read_text())
    pilot_receipt=json.loads((pilot_review/'receipt.json').read_text())
    if (baseline_receipt['download_sha256']['teacher_pair_base_v2/evaluation.json']!=file_sha(baseline)
            or baseline_receipt['packet_manifest_sha256']!=file_sha(control/'manifest.json')
            or pilot_receipt['packet_manifest_sha256']!=file_sha(control/'manifest.json')
            or pilot_receipt['external_baseline_sha256']!=file_sha(baseline)):
        raise ValueError('reviewed_control_baseline_changed')
    prior=json.loads((pilot_review/'summary.json').read_text())
    if prior.get('execution_complete') is not True or prior.get('optimizer_steps')!=46:
        raise ValueError('completed_original_input_control_required')
    original=[json.loads(line) for line in old_path.read_text().splitlines()]
    selected=[json.loads(line) for line in (anchors/'anchors.private.jsonl').read_text().splitlines()]
    records,audit=replace_inputs(original,selected)
    replaced=sorted(a['example_id'] for a in audit)
    if set(replaced)&set(control_config['training_probe_ids']):raise ValueError('keep_both_training_probes_unchanged')
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir();(output/'private_key').mkdir()
    data=upload/'teacher_pair_replacement.jsonl';raw=old_path.read_bytes().splitlines(keepends=True)
    with data.open('wb') as stream:
        for old_line,row in zip(raw,records):
            stream.write((json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n').encode()
                         if row['example_id'] in replaced else old_line)
    config=copy.deepcopy(control_config)
    config.update(protocol=runtime.PROTOCOL,dataset_sha256=file_sha(data),control_config=control_config,
        control_baseline_sha256=file_sha(baseline),control_runtime_sha256=file_sha(Path(control_runtime.__file__)),
        fresh_adapter_required=True,replaced_training_ids=replaced)
    cfg=upload/'teacher_pair_replacement_config.json';write_json(cfg,config)
    runtime.load_inputs(data,cfg);runtime.validate_baseline(baseline,config)
    write_rows(output/'private_key/replacement_audit.private.jsonl',audit)
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','training_pairs':46,'development_pairs':8,
        'replaced_training_inputs':12,'unchanged_training_inputs':34,'training_targets_unchanged':46,
        'development_row_bytes_unchanged':True,'original_row_order_unchanged':True,'target_frequencies_unchanged':True,
        'families':dict(Counter(a['family'] for a in audit)),
        'epochs':2,'optimizer_updates':46,'updates_per_epoch':23,'fresh_adapter_from_base':True,
        'training_authorized_in_initial_packet':False,'gpu_jobs_submitted':0,'reserved_test_rows_used':0,
        'generation_instruction_unchanged':True,'training_algorithm_and_hyperparameters_unchanged':True,
        'base_snapshot_generations':10,'total_generations_including_reload':31,
        'baseline_reuse':'Exact reviewed expanded baseline; common base generations/losses must match before gradients.',
        'control_baseline_sha256':file_sha(baseline),'control_baseline_slug':'oliverlds/teacher-pair-expanded-base-4b',
        'control_baseline_version':1,'control_pilot_slug':'oliverlds/teacher-pair-expanded-pilot-4b','control_pilot_version':1,
        'control_pilot_review_manifest_sha256':file_sha(pilot_review/'manifest.json'),
        'control_packet_manifest_sha256':file_sha(control/'manifest.json'),'anchor_manifest_sha256':file_sha(anchors/'manifest.json'),
        'dataset_sha256':file_sha(data),'config_sha256':file_sha(cfg),'runtime_source_sha256':file_sha(Path(runtime.__file__)),
        'tokenizer_preflight':'Kaggle checks all 54 rows before weights; pending for this exact run',
        'comparison_limit':'Single-seed matched input-replacement ablation on reused development cases, not generalization or a Fama-style benchmark.'}
    write_json(output/'plan.json',plan)
    write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,'builder_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('control','anchors','baseline','baseline-review','pilot-review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.control,args.anchors,args.baseline,args.baseline_review,args.pilot_review,args.output)))


if __name__=='__main__':main()
