"""CPU-only signal diagnosis, constrained inference and training-data preparation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from .claims import sha, risk_flags
from .signal import fit_profile, diagnose, overlap, distance
from .prototype import rewrite
from .experiments import prepare, plans

ROOT=Path(__file__).resolve().parents[3]
EXPECTED={"train":"a108416c1935615a7e01bb4d3fa86e0818b961bfb16f15d3883e6905654ee248",
          "validation":"f1c75c29fa93f8458e7a3b2f8996bbd13ff023f92e4a74f1d086d22af5e546b8"}


def read_split(path, split):
    if split not in EXPECTED or Path(path).name=='test.jsonl':
        raise ValueError('Reserved test input is forbidden')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED[split]:raise ValueError('Frozen '+split+' hash mismatch')
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if any(row['split']!=split for row in rows):raise ValueError('Mixed split input')
    return rows


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_lines(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def synthetic():
    # Original synthetic examples; no paper text or model outputs.
    inputs=[('We use data to measure output. Firms report revenue.','We use data to measure output; Firms report revenue.'),
            ('If demand falls, firms reduce output.','Firms reduce output if demand falls.'),
            ('Some firms may not increase investment. We find no effect.','Some firms may not increase investment. We find no effect.')]
    train=[{'id':f'synthetic-train-{i}','paper_key':'SYN','split':'train','plain_english':a,'completion':b,'prompt':a}
           for i,(a,b) in enumerate(inputs)]
    validation=[{'id':'synthetic-validation','paper_key':'SYN-HELD','split':'validation',
                 'plain_english':'Firms report sales. Investors use the reports.',
                 'completion':'Firms report sales; Investors use the reports.'}]
    return train,validation


def execute(train, validation, directory, synthetic_run=False):
    if any(row['split']!='train' for row in train) or any(row['split']!='validation' for row in validation):
        raise ValueError('Split leakage detected')
    if set(row['id'] for row in train)&set(row['id'] for row in validation):raise ValueError('Overlapping IDs')
    if directory.exists() and any(directory.iterdir()):raise ValueError('Refusing to overwrite research outputs')
    directory.mkdir(parents=True,exist_ok=True)
    profile=fit_profile(train)
    pair_diagnostics,phase_a=diagnose(train,profile)
    write_json(directory/'A_train_signal_summary.json',phase_a)
    write_json(directory/'A_style_profile.json',profile)
    write_lines(directory/'A_pair_diagnostics.private.jsonl',pair_diagnostics)
    outputs=[]
    for row in validation:
        result=rewrite(row['plain_english'],profile)
        outputs.append({'id':row['id'],**result,
                        'reference_risk_screen':risk_flags(row['plain_english'],row['completion'])})
    write_lines(directory/'B_validation_prototype.private.jsonl',outputs)
    phase_b={'validation_rows':len(validation),'generated_candidates':sum(len(r['candidates']) for r in outputs),
             'surface_accepted_candidates':sum(c['verification']['status']=='surface_preserved' for r in outputs for c in r['candidates']),
             'selected_rewrites':sum(r['status']=='selected_bounded_rewrite' for r in outputs),
             'abstained_unchanged':sum(r['status']=='abstained_unchanged' for r in outputs),
             'mean_selected_source_overlap':sum(overlap(row['plain_english'],r['selected_text']) for row,r in zip(validation,outputs))/len(outputs),
             'mean_selected_style_gain_proxy':sum(distance(row['plain_english'],profile)-distance(r['selected_text'],profile)
                                                   for row,r in zip(validation,outputs))/len(outputs),
             'model_generation_executed':False,'human_judgment_used':False,
             'semantic_success_rate':None,'semantic_success_reason':'No calibrated semantic parser/NLI; bounded surface checks do not establish general entailment',
             'test_rows_used':0}
    write_json(directory/'B_prototype_summary.json',phase_b)
    training=prepare(train,profile)
    for name,key in (('C_A_sft_candidates.private.jsonl','sft'),('C_B_weak_preferences.private.jsonl','preferences'),
                     ('C_B_quarantined_preferences.private.jsonl','quarantined_preferences'),('C_parent_coverage.private.jsonl','coverage')):
        write_lines(directory/name,training[key])
    write_json(directory/'C_preparation_summary.json',training['summary'])
    write_json(directory/'C_experiment_plans.json',plans())
    summary={'run_kind':'synthetic_dry_run' if synthetic_run else 'executed_cpu_research',
             'phase_a':phase_a,'phase_b':phase_b,'phase_c':training['summary'],
             'gpu_jobs_submitted':0,'gpu_training_executed':False,'test_rows_used':0,
             'existing_experiments_modified':False}
    write_json(directory/'summary.json',summary)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    write_json(directory/'manifest.json',{'implementation':'claim_style_extension_v1',
        'inputs':{'training_rows':len(train),'validation_rows':len(validation),'training_split_sha256':None if synthetic_run else EXPECTED['train'],
                  'validation_split_sha256':None if synthetic_run else EXPECTED['validation']},
        'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'files':files,'no_model_or_api_calls':True,'test_rows_used':0})
    print(json.dumps({'run_kind':summary['run_kind'],'train_rows':len(train),'validation_rows':len(validation),
                      'selected_prototype_rewrites':phase_b['selected_rewrites'],
                      'sft_candidates':training['summary']['sft_variants_staged'],
                      'training_ready':False,'gpu_jobs_submitted':0,'output_dir':str(directory)}))
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Use only synthetic examples; CPU, no model/API')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    train,validation=synthetic() if args.dry_run else (read_split(ROOT/'data/fama_corpus_v1/train.jsonl','train'),
                                                   read_split(ROOT/'data/fama_corpus_v1/validation.jsonl','validation'))
    execute(train,validation,args.output_dir,args.dry_run)


if __name__=='__main__':main()
