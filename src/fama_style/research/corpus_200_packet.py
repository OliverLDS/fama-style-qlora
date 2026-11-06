"""Freeze the 200-pair pilot inputs and sixteen development cases; CPU only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from fama_style.training import corpus_200_pilot as runtime


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def write_rows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def verify_parent(path):
    manifest=json.loads((path.parent/'manifest.json').read_text())
    expected=manifest.get('files',manifest.get('outputs',{})).get(path.name)
    if expected!=file_sha(path):raise ValueError('changed_parent_dataset')


def build(training,development,output):
    if output.exists():raise ValueError('refuse_overwrite_frozen_pilot_packet')
    verify_parent(training);verify_parent(development)
    train=read_rows(training);dev=read_rows(development)
    if len(train)!=200:raise ValueError('frozen_200_pair_cohort_required')
    for split,rows in [('training',train),('development',dev)]:
        if any(r['split']!=split or r['quality']['text_pair_status']!='automated_pilot_approved' for r in rows):
            raise ValueError('only_approved_rows_in_existing_roles')
    if Counter(r['style_group'] for r in train)!=Counter({g:50 for g in runtime.GROUPS}):
        raise ValueError('fifty_pairs_per_group_required')
    train_papers={r['paper_id'] for r in train}
    if train_papers&{r['paper_id'] for r in dev}:raise ValueError('paper_leakage')
    selected=[];papers_by_group={};rank=lambda s:runtime.sha('corpus200-dev-v1|'+s)
    for group in runtime.GROUPS:
        papers=defaultdict(list)
        for row in dev:
            if row['style_group']==group:papers[row['paper_id']].append(row)
        eligible=sorted((p for p in papers if len(papers[p])>=2),key=rank)
        if len(eligible)<2:raise ValueError('two_eligible_development_papers_per_group_required')
        papers_by_group[group]=eligible[:2]
        for paper in eligible[:2]:selected.extend(sorted(papers[paper],key=lambda r:rank(r['id']))[:2])
    (output/'upload').mkdir(parents=True);(output/'private_key').mkdir()
    export=[]
    for split,rows in [('training',train),('development',selected)]:
        for r in rows:
            export.append({'example_id':r['id'],'parent_id':r['parent_id'],'paper_id':r['paper_id'],
                'style_group':r['style_group'],'authors':r['authors'],'split':split,
                'source':r['plain_english'],'target':r['original_paragraph'],
                'source_sha256':r['source_sha256'],'target_sha256':r['target_sha256'],
                'prompt':runtime.prompt(r['plain_english']),'pilot_eligible':True})
    data=output/'upload'/'corpus_200_pairs.jsonl';write_rows(data,export)
    # Eight cases: two examples from one prespecified development paper per
    # author group. Cost and PDF packets are computed from actual outputs later.
    detector_ids=[r['id'] for g in runtime.GROUPS for r in selected
                  if r['style_group']==g and r['paper_id']==papers_by_group[g][0]]
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        'hyperparameters':runtime.HYPER,'decoding':runtime.DECODING,'packages':runtime.PACKAGES,
        'max_runtime_seconds':runtime.MAX_SECONDS,'training_authorized':True,
        'quantization':'NF4 double quantization; FP32 compute',
        'runtime_source_sha256':file_sha(Path(runtime.__file__)),'dataset_sha256':file_sha(data),
        'training_ids':[r['id'] for r in train],'development_ids':[r['id'] for r in selected],
        'training_probe_ids':[min((r for r in train if r['style_group']==g),key=lambda r:rank(r['id']))['id'] for g in runtime.GROUPS],
        'detector_case_ids':detector_ids,'training_cohort_sha256':file_sha(training),
        'development_parent_sha256':file_sha(development),
        'authorization_scope':{'user_instruction':'do it','epochs':1,'updates':100,'training_pairs':200,
            'development_pairs':16,'matched_generations':32,'GPU':'one Kaggle T4','max_seconds':5400,
            'automatic_submission':False,'automatic_retries':False,'full_training':False},
        'evaluation_limits':'Paper-held-out within known author groups; reused development exposure; automated QA, not independent gold. No Fama-specific or detector success claimed.'}
    path=output/'upload'/'corpus_200_config.json';runtime.write_json(path,config)
    runtime.load_inputs(data,path)
    write_rows(output/'private_key'/'development_lineage.private.jsonl',selected)
    summary={'protocol':runtime.PROTOCOL,'status':'prepared_not_gpu_executed','training_pairs':200,
             'training_groups':{g:50 for g in runtime.GROUPS},'training_papers':len(train_papers),
             'development_pairs':16,'development_papers':8,'development_group_counts':{g:4 for g in runtime.GROUPS},
             'detector_cases_preselected':8,'epochs':1,'optimizer_steps':100,'matched_generations':32,
             'max_runtime_seconds':5400,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
             'base_initialization':'Fresh pinned base, not a continuation of earlier adapters',
             'root_1b_pipeline_modified':False,'reserved_examples_accessed':0,'training_cohort_sha256':file_sha(training),
             'GPU_jobs_submitted':0,'external_model_calls':0,'dependency_basis':'Same exact pins as the previously successful expanded teacher-pair runtime; no upgrade experiment.'}
    runtime.write_json(output/'plan.json',summary)
    runtime.write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,
        'inputs':{str(training):file_sha(training),str(development):file_sha(development)},
        'runtime_source_sha256':config['runtime_source_sha256'],
        'builder_source_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary
