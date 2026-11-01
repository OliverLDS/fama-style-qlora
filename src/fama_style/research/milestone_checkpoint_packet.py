"""Reconcile the frozen sparse subset and reuse compatible ordinary 4B outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from fama_style.evaluation import milestone_checkpoint_4b as runtime
from .author_control_replication import verify_files
from .multi_author_milestone import make_job,profile,INSTRUCTION
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def select_frozen(path,subset):
    ids={r['example_id'] for r in subset};selected={}
    for line in path.read_text().splitlines():
        if not any('"'+rid+'"' in line for rid in ids):continue
        row=json.loads(line)
        if row['example_id'] in ids:selected[row['example_id']]=row
    if len(subset)!=8 or len(ids)!=8 or set(selected)!=ids:raise ValueError('exact_frozen_eight_required')
    if len({r['paper_id'] for r in subset})!=4 or any(n!=2 for n in Counter(r['paper_id'] for r in subset).values()):
        raise ValueError('two_cases_per_four_papers_required')
    if any(Counter(r['paper_id'] for r in subset if r['round']==i)!=Counter({p:1 for p in {r['paper_id'] for r in subset}}) for i in (1,2)):
        raise ValueError('original_two_paper_balanced_rounds_required')
    for meta in subset:
        row=selected[meta['example_id']]
        if (row['split']!='development' or any(row[k]!=meta[k] for k in ('paper_id','source_sha256','target_sha256'))
                or row['source_sha256']!=runtime.sha(row['source']) or row['target_sha256']!=runtime.sha(row['target'])
                or ' '.join(' '.join(row['source_sentences']).split())!=' '.join(row['source'].split())):
            raise ValueError('frozen_case_text_or_source_sentence_coverage_changed')
    return [selected[r['example_id']] for r in subset]


def reuse_generation(job,generation,reference):
    if (generation['example_id']!=job['example_id'] or generation['source']!=job['source']
            or generation['reference']!=reference or generation['prompt_sha256']!=job['prompt_sha256']):
        raise ValueError('archived_output_not_compatible_with_checkpoint_job')
    return {k:job[k] for k in ('job_id','example_id','condition','source','source_sha256','prompt_sha256')}|{
        k:generation[k] for k in ('output','output_tokens','empty','output_limited','copied_source')}|{'origin':'archived'}


def build(parent,sparse,training,base_download,base_review,pilot_download,pilot_review,output):
    if output.exists():raise ValueError('refuse_overwrite_checkpoint_packet')
    for folder in (parent,sparse,training,base_review,pilot_review):verify_files(folder)
    if INSTRUCTION!=runtime.INSTRUCTION:raise ValueError('ordinary_instruction_must_match_prior_protocol')
    subset=json.loads((sparse/'PRIVATE_EVALUATION_SUBSET.json').read_text())
    selected=select_frozen(parent/'datasets/development.private.jsonl',subset)
    train=[json.loads(l) for l in (training/'training.private.jsonl').read_text().splitlines()]
    if len(train)!=46 or any(r['split']!='training' for r in train):raise ValueError('actual_46_training_targets_only')
    if {a.casefold() for r in train for a in r['authors']}&{a.casefold() for r in selected for a in r['authors']}:
        raise ValueError('author_split_overlap')
    style=profile(train)
    base_path=base_download/'teacher_pair_base_v2/evaluation.json';pilot_path=pilot_download/'teacher_pair_pilot_v2/evaluation.json'
    base=json.loads(base_path.read_text());pilot=json.loads(pilot_path.read_text())
    br=json.loads((base_review/'receipt.json').read_text());pr=json.loads((pilot_review/'receipt.json').read_text())
    if (br['download_sha256']['teacher_pair_base_v2/evaluation.json']!=file_sha(base_path)
            or pr['download_files_sha256']['teacher_pair_pilot_v2/evaluation.json']!=file_sha(pilot_path)
            or not base['complete'] or not pilot['complete'] or base['config']!=pilot['config']
            or base['runtime']['model_id']!=runtime.MODEL_ID or base['runtime']['model_revision']!=runtime.REVISION
            or base['runtime']['packages']!=pilot['runtime']['packages']
            or base['config']['decoding']!=runtime.DECODING or base['config']['hyperparameters']['seed']!=1729):
        raise ValueError('reviewed_archive_settings_changed')
    pools={'base_ordinary':{r['example_id']:r for r in base['snapshots'][0]['generations']},
           'adapter_ordinary':{r['example_id']:r for r in pilot['snapshots'][2]['generations']}}
    token_counts={r['example_id']:r['masked_tokens'] for r in json.loads((base_download/'teacher_pair_base_v2/label_audit.json').read_text())}
    if br['download_sha256']['teacher_pair_base_v2/label_audit.json']!=file_sha(base_download/'teacher_pair_base_v2/label_audit.json'):
        raise ValueError('archived_token_audit_changed')
    if br['download_sha256']['teacher_pair_base_v2/generation_config.json']!=file_sha(base_download/'teacher_pair_base_v2/generation_config.json'):
        raise ValueError('archived_generation_config_changed')
    jobs=[];reused=[];reuse_tokens={};reconciliation=[]
    for row,meta in zip(selected,subset):
        for condition in runtime.CONDITIONS:
            old_condition='source_structural_prior' if condition=='base_source_structural_prior' else 'ordinary'
            original_job=make_job(row,style,old_condition)
            job={'job_id':row['example_id']+'__'+condition,'example_id':row['example_id'],'condition':condition,
                'source':row['source'],'source_sha256':row['source_sha256'],
                'prompt':original_job['prompt'],'prompt_sha256':original_job['prompt_sha256'],'run_required':True}
            archived=pools.get(condition,{}).get(row['example_id'])
            if archived is not None:
                job['run_required']=False;reused.append(reuse_generation(job,archived,row['target']))
                reuse_tokens[job['job_id']]=token_counts[row['example_id']]
            jobs.append(job)
        reconciliation.append({'example_id':row['example_id'],'paper_id':row['paper_id'],'round':meta['round'],
            'source_sha256':row['source_sha256'],'target_sha256':row['target_sha256'],
            'in_current_teacher_development':row['example_id'] in pools['base_ordinary'],
            'original_needs_review':row['needs_review'],'needs_external_context':row['needs_external_context'],
            'source_quality_status':row['source_quality_status']})
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir();(output/'private_key').mkdir()
    jobs_path=upload/'milestone_checkpoint_jobs.jsonl';write_rows(jobs_path,jobs)
    adapter=pilot_download/'teacher_pair_pilot_v2/epoch_2/adapter'
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        'jobs_sha256':file_sha(jobs_path),'example_ids':sorted(r['example_id'] for r in selected),
        'seed':1729,'context_limit':4096,'decoding':runtime.DECODING,'quantization':'NF4 double quantization; FP32 compute',
        'expected_packages':base['runtime']['packages'],'training_authorized':False,
        'expected_generation_config':json.loads((base_download/'teacher_pair_base_v2/generation_config.json').read_text()),
        'reused_prompt_tokens':reuse_tokens,'adapter_sha256':{name:file_sha(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}}
    for name,digest in config['adapter_sha256'].items():
        if digest!=pr['download_files_sha256']['teacher_pair_pilot_v2/epoch_2/adapter/'+name]:raise ValueError('reviewed_adapter_changed')
    cfg=upload/'milestone_checkpoint_config.json';write_json(cfg,config);runtime.load_inputs(jobs_path,cfg)
    write_rows(output/'private_key/reference_key.private.jsonl',[{k:r[k] for k in ('example_id','paper_id','authors','source','target','source_sha256','target_sha256','needs_review','needs_external_context')} for r in selected])
    write_rows(output/'private_key/reused_outputs.private.jsonl',reused);write_json(output/'private_key/reconciliation.private.json',reconciliation)
    write_json(output/'private_key/frozen_subset.json',subset);write_json(output/'training_style_profile.json',style)
    rubric={'version':'checkpoint_drift_rubric_v1','assess_against_source_first':True,
        'meaning_equivalent_rewordings_count_as_drift':False,'minor_semantic_shift_budget_per_output':1,
        'material_shifts_allowed_for_eligible_output':0,'unresolved_is_success':False,
        'minor_examples':['small qualifier weakening without reversing core inference','nonessential illustrative or roadmap omission','mild editorial qualification without new economic relation'],
        'material_examples':['changed actor/role or economic quantity','reversed chronology or causal relation','new explanatory mechanism','lost central condition, negation or substantive supporting claim'],
        'aggregate_progress_gate':{'detector':'lower native flag rate than ordinary base, only if endpoint informative',
            'meaning':'no increase in material-shift rate; local-budget compliance and unresolved cases reported',
            'readability_expression':'paired wins exceed losses','copying':'no increase in exact input-copy rate'},
        'not_a_fama_style_benchmark':True,'assessment_limits':'Versioned prospective rubric for the missing outputs; historical outputs already known; same-reviewer automated judgments are provisional.'}
    write_json(output/'quality_rubric.json',rubric)
    policy=json.loads((sparse/'protocol.json').read_text());write_json(output/'detector_policy.json',policy)
    known_words=sum(len(r['source'].split())+len(r['target'].split()) for r in selected)+sum(len(r['output'].split()) for r in reused)
    pending_estimate=sum(len(next(r['target'] for r in selected if r['example_id']==j['example_id']).split()) for j in jobs if j['run_required'])
    calibration=json.loads((sparse/'summary.json').read_text())['initial_matched_pair_words_estimate']
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','frozen_cases':8,'development_papers':4,'cases_per_paper':2,
        'overlap_with_recent_eight':sum(r['in_current_teacher_development'] for r in reconciliation),
        'reused_outputs':10,'new_outputs_planned':14,'pending_by_arm':dict(Counter(j['condition'] for j in jobs if j['run_required'])),
        'model_arms':list(runtime.CONDITIONS),'ordinary_instruction_unchanged':True,'profile_training_pairs':46,
        'structural_prior':'Same optional <=30-word merge algorithm and source-sentence display; pooled profile refit on actual 46 training targets. Not a learned mapper or oracle.',
        'context_policy':'4096 guard for longer structural prompts; ordinary jobs remain within the prior 1024 guard. No RoPE change or truncation.',
        'planned_detector_slots':40,'known_text_words':known_words,'missing_model_text_words_estimate':pending_estimate,
        'initial_calibration_pair_words_estimate':calibration,'estimated_project_words':known_words+pending_estimate+calibration,
        'detector_max_project_words':policy['target_max_project_words'],'actual_output_words_and_account_balance_known':False,
        'detector_calls_made':0,'training_authorized':False,'optimizer_steps':0,'gpu_jobs_submitted':0,
        'reserved_records_used':0,'runtime_source_sha256':file_sha(Path(runtime.__file__)),
        'jobs_sha256':file_sha(jobs_path),'config_sha256':file_sha(cfg),
        'adapter_slug':'oliverlds/teacher-pair-expanded-pilot-4b','adapter_version':1,
        'lineage_sha256':{'parent_manifest':file_sha(parent/'manifest.json'),'sparse_manifest':file_sha(sparse/'manifest.json'),
            'training_manifest':file_sha(training/'manifest.json'),'base_review_manifest':file_sha(base_review/'manifest.json'),
            'pilot_review_manifest':file_sha(pilot_review/'manifest.json')},
        'limits':['Three reintroduced cases retain original input/reference/context flags; no source repair or fresh-test claim.',
            'Archived versus new sessions distinguished; exact settings/token counts checked but no extra reproduction generations.',
            'Structural prior changes prompt length and repeats source sentences, so it does not isolate abstract structure.',
            'Sparse detector endpoint is not yet shown informative; unflagged GPT-6 control alone does not establish Gemma-output floor.']}
    write_json(output/'plan.json',plan)
    write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,'builder_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','sparse','training','base-download','base-review','pilot-download','pilot-review','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.parent,a.sparse,a.training,a.base_download,a.base_review,a.pilot_download,a.pilot_review,a.output)))


if __name__=='__main__':main()
