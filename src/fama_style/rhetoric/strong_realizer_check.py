"""Manual external-realizer diagnostic: exact frozen inner prompts, CPU import only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import statistics
from . import operation_realization as parent
from .concise_plan import file_sha,read_lines
from .experiment import fresh,write_json,write_lines
from fama_style.research.claims import sha,risk_flags
from fama_style.research.relation_guards import check as relation_check

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='strong_realizer_check_v1'
CONDITIONS=(parent.CONDITIONS[0],parent.CONDITIONS[2])
RESPONSE_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','type':'object',
    'properties':{'job_id':{'type':'string','pattern':'^J[0-9a-f]{16}$'},'prompt_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'output':{'type':'string','minLength':1}},
    'required':['job_id','prompt_sha256','output'],'additionalProperties':False}
TASK_PROMPT='''Process the attached jobs_01.jsonl or jobs_02.jsonl as independent rewriting tasks.

For each job, follow its prompt. Do not alter its source or plan, borrow facts
from another job, retrieve reference text, or use external browsing. Produce one
candidate per job, without selecting among retries. Do not grade the outputs or
add critiques, claim ledgers, commentary or suggested improvements.

Delivery contract: the inner prompt's request to return only the paragraph
applies to the content of the output field. Deliver one JSON object per line,
with exactly these fields:
{"job_id":"COPY_FROM_INPUT","prompt_sha256":"COPY_FROM_INPUT","output":"YOUR_REWRITTEN_PARAGRAPH"}

Copy job_id and prompt_sha256 exactly. Do not invent a model version or a
completion claim. Return a downloadable UTF-8 JSONL file without Markdown
fences: responses_01.jsonl for jobs_01.jsonl, responses_02.jsonl for jobs_02.jsonl.
If completion is impossible, return completed rows only and state the missing
job IDs separately. Do not fabricate an output for an incomplete job.
'''
START_HERE='''# Manual external-realizer comparison — prepared, not executed

Use the same chosen model and visible settings for both batches. This package
contains 10 outputs to generate, not 10 completed results.

1. Start a new conversation and attach ONLY batches/jobs_01.jsonl and
   response.schema.json. Paste TASK_PROMPT.md as your request.
2. In a separate new conversation with the same chosen model, attach ONLY
   batches/jobs_02.jsonl and response.schema.json. Paste the same TASK_PROMPT.md.
3. Save responses_01.jsonl and responses_02.jsonl in:
   research_runs/strong_realizer_check_v1/responses/
4. Optionally save notes.txt there with the visible model label, known settings,
   and whether the two batches used separate conversations. Unknown is fine.
   Do not ask the model to infer its exact underlying version or decoding seed.
5. Tell Codex the files are ready for CPU import and assessment.

Do NOT attach private_key.private.json, manifest.json, previous outputs,
reference paragraphs, reports, or this START_HERE file to the generator.
Only the two batch files, shared task prompt and response schema are generation
materials. They contain private source text: do not publish them or include any
credentials. No API key, HF_TOKEN or Kaggle session is needed for this package.

The batches carry the exact frozen ordinary and operation-guided inner prompts
from the completed Kaggle diagnostic, with the same sources and source-evidence
scaffold. Which batch is which is recorded in the private key. All five cases
are previously inspected training examples; one is a preservation control.
There is no reference wording in the generation files, but the structural plans
remain reference-informed oracle information.

The model receives five jobs per chat and a JSONL delivery wrapper, unlike the
single-job greedy Kaggle calls. Hidden decoding, token budgets, model identity,
chat memory, possible source familiarity and batch interactions are uncontrolled.
New conversations reduce direct shared answer context; they do not prove
independence, and clearing chats/memory is not required. Do not claim a matched
model-size experiment or held-out generalization result.

Keep the first returned files. Do not edit an output, paraphrase it again,
select a better retry or silently fix malformed JSON. Partial/invalid records
will be retained and reported. The importer checks structure, hashes, coverage,
copying and bounded surface alerts; it cannot prove semantic fidelity or author
style. No human-review task is required.
'''


def loads_unique(text):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate_json_key')
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=unique)


def prepare(output):
    root=ROOT/'research_runs/grounded_operation_realization_v1'
    release=json.loads((root/'release_verification.json').read_text())
    packpath=root/'input_pack/operation_pack.json';pack=parent.load_pack(packpath,release['pack_sha256'])
    review=json.loads((root/'latest_review_v1/verification.json').read_text())
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Parent run review incomplete')
    run=root/'kaggle_latest/grounded_operation_realization_v1'
    if file_sha(run/'manifest.json')!=review['artifact_hashes']['manifest.json']:raise ValueError('Parent run changed')
    jobs=[j for j in parent.jobs_for(pack) if j['model']=='base' and j['condition'] in CONDITIONS]
    private_jobs=[];batches={};fresh(output);(output/'batches').mkdir()
    for batch,condition in enumerate(CONDITIONS,1):
        filename=f'jobs_{batch:02d}.jsonl';requests=[]
        # Same deterministic paragraph order in both chats; labels remain opaque.
        for job in sorted((j for j in jobs if j['condition']==condition),key=lambda j:sha(j['id'])):
            job_id='J'+sha(PROTOCOL+'|'+job['id']+'|'+condition)[:16]
            record={'job_id':job_id,'prompt_sha256':job['prompt_sha256'],'prompt':job['prompt']}
            requests.append(record)
            private_jobs.append({'job_id':job_id,'source_id':job['id'],'condition':condition,'use':job['use'],
                'batch_file':filename,'response_file':f'responses_{batch:02d}.jsonl','prompt_sha256':job['prompt_sha256']})
        write_lines(output/'batches'/filename,requests);batches[filename]=file_sha(output/'batches'/filename)
    write_json(output/'private_key.private.json',{'protocol':PROTOCOL,'parent_pack_sha256':release['pack_sha256'],
        'parent_manifest_sha256':file_sha(run/'manifest.json'),'jobs':private_jobs,'rows':pack['rows'],'style_profile':pack['style_profile'],
        'model_identity':'not_supplied_not_verified','run_status':'prepared_not_executed'})
    (output/'TASK_PROMPT.md').write_text(TASK_PROMPT,encoding='utf-8')
    (output/'START_HERE.md').write_text(START_HERE,encoding='utf-8')
    write_json(output/'response.schema.json',RESPONSE_SCHEMA)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','code_sha256':file_sha(Path(__file__)),
        'parent_pack_sha256':release['pack_sha256'],'parent_manifest_sha256':file_sha(run/'manifest.json'),
        'requested_outputs':10,'contrast_outputs':8,'preservation_control_outputs':2,'model_calls':0,'training_called':False,
        'validation_rows_read':0,'reserved_test_rows_read':0,
        'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Prepared two five-job external-generation batches, with exact frozen inner prompts. No external request or GPU run made.')


def check_package(folder,verify_parent=True):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['code_sha256']!=file_sha(Path(__file__)):raise ValueError('Package/compiler identity changed')
    for name,h in manifest['files'].items():
        if file_sha(folder/name)!=h:raise ValueError('Prepared file changed')
    key=json.loads((folder/'private_key.private.json').read_text());expected={j['job_id']:j for j in key['jobs']};seen=set()
    if len(expected)!=10 or len(key['jobs'])!=10:raise ValueError('Expected ten unique jobs')
    for batch in sorted((folder/'batches').glob('*.jsonl')):
        records=read_lines(batch)
        if len(records)!=5:raise ValueError('Expected five jobs per batch')
        for r in records:
            j=expected.get(r['job_id'])
            if set(r)!={'job_id','prompt_sha256','prompt'} or not j or j['batch_file']!=batch.name or r['job_id'] in seen or sha(r['prompt'])!=r['prompt_sha256'] or j['prompt_sha256']!=r['prompt_sha256']:raise ValueError('Changed job or leaked metadata')
            seen.add(r['job_id'])
    if seen!=set(expected):raise ValueError('Missing jobs')
    if verify_parent:
        root=ROOT/'research_runs/grounded_operation_realization_v1';path=root/'input_pack/operation_pack.json'
        pack=parent.load_pack(path,key['parent_pack_sha256']);original={(j['id'],j['condition']):j for j in parent.jobs_for(pack) if j['model']=='base' and j['condition'] in CONDITIONS}
        for batch in (folder/'batches').glob('*.jsonl'):
            for r in read_lines(batch):
                j=expected[r['job_id']]
                if r['prompt']!=original[(j['source_id'],j['condition'])]['prompt']:raise ValueError('Inner prompt differs from frozen experiment')
    return key


def parse_responses(key,folder):
    expected={j['job_id']:j for j in key['jobs']};candidates=[];errors=[];observed=[];input_hashes={}
    for name in sorted({j['response_file'] for j in key['jobs']}):
        path=folder/name
        if not path.exists():errors.append({'file':name,'code':'missing_response_file'});continue
        input_hashes[name]=file_sha(path)
        try:text=path.read_text(encoding='utf-8')
        except UnicodeError:errors.append({'file':name,'code':'invalid_utf8'});continue
        for line_number,line in enumerate(text.splitlines(),1):
            if not line.strip():continue
            location={'file':name,'line':line_number}
            try:r=loads_unique(line)
            except (ValueError,TypeError):errors.append({**location,'code':'invalid_json_or_duplicate_keys'});continue
            if not isinstance(r,dict):errors.append({**location,'code':'not_an_object'});continue
            if isinstance(r.get('job_id'),str) and r['job_id'] in expected:observed.append(r['job_id'])
            if set(r)!={'job_id','prompt_sha256','output'}:errors.append({**location,'code':'wrong_fields'});continue
            j=expected.get(r['job_id']) if isinstance(r['job_id'],str) else None
            if not j:errors.append({**location,'code':'unknown_job'});continue
            if j['response_file']!=name:errors.append({**location,'code':'wrong_batch'});continue
            if r['prompt_sha256']!=j['prompt_sha256']:errors.append({**location,'code':'prompt_hash_mismatch'});continue
            if not isinstance(r['output'],str) or not r['output'].strip():errors.append({**location,'code':'empty_or_nontext_output'});continue
            candidates.append({**r,'provenance':location})
    duplicate_ids={j for j,n in Counter(observed).items() if n>1}
    for job_id in sorted(duplicate_ids):errors.append({'job_id':job_id,'code':'duplicate_job_all_copies_excluded'})
    valid=[r for r in candidates if r['job_id'] not in duplicate_ids]
    missing=sorted(set(expected)-{r['job_id'] for r in valid})
    return valid,{'complete':not errors and not missing,'expected_jobs':len(expected),'accepted_jobs':len(valid),
        'missing_or_rejected_job_ids':missing,'errors':errors,'response_file_hashes':input_hashes,'notes_sha256':file_sha(folder/'notes.txt') if (folder/'notes.txt').exists() else None,
        'no_repairs_or_best_of_selection':True}


def citation_context_alerts(source,output):
    # Fixed-case surface warning for the control's observed error. Not a semantic
    # attribution verifier: a legitimate contrast can put both topics in a sentence.
    if not all(name in source for name in ('Roll (1986)','Mitchell and Stafford (1997)')):return []
    alerts=[]
    for i,sentence in enumerate(parent.grouping.output_units(output)['units'],1):
        for name in ('Roll','Mitchell'):
            if re.search(r'\b'+name+r'\b',sentence) and re.search(r'proxy|contest',sentence,re.I):
                alerts.append({'kind':'citation_in_proxy_context_surface_alert','author':name,'output_sentence':i,
                    'semantic_error_verified':False,'limitation':'Same-sentence co-occurrence only; may flag a legitimate contrast.'})
    return alerts


def audit_responses(package,responses,output):
    key=check_package(package);valid,coverage=parse_responses(key,responses);fresh(output);(output/'raw').mkdir()
    for name in coverage['response_file_hashes']:shutil.copyfile(responses/name,output/'raw'/name)
    jobs={j['job_id']:j for j in key['jobs']};rows={r['id']:r for r in key['rows']};metrics=[]
    for r in valid:
        j=jobs[r['job_id']];row=rows[j['source_id']];out={'output':r['output'],'hit_output_limit':False}
        metric=parent.previous.output_metrics({'style_profile':key['style_profile']},{'source':row['source'],'reference':row['reference'],'plan':row['shallow_plan']},out)
        # The transport does not report a tokenizer cutoff; do not pretend otherwise.
        metric.pop('hit_output_limit');metric.pop('available')
        metrics.append({**metric,'job_id':r['job_id'],'source_id':row['id'],'condition':j['condition'],'use':j['use'],
            'prompt_sha256':j['prompt_sha256'],'provenance':r['provenance'],'model_identity':'not_verified',
            'termination_reason':'not_reported','semantic_assessment':'not_executed','operation_realization':'not_assessed',
            'relation_surface_checks':relation_check(row['source'],r['output']),
            'citation_context_alerts':citation_context_alerts(row['source'],r['output'])})
    write_json(output/'coverage.json',coverage);write_lines(output/'metrics.private.jsonl',metrics)
    summary={'protocol':PROTOCOL,'status':'import_complete_quality_assessment_pending' if coverage['complete'] else 'partial_or_invalid_import_quality_assessment_pending',
        'expected_outputs':len(key['jobs']),'accepted_outputs':len(valid),'conditions':{},'semantic_accuracy':None,'author_style_accuracy':None,
        'model_identity':'not_verified','matched_decoding':False,'batch_context_matches_kaggle':False,
        'model_calls':0,'gpu_jobs_submitted':0,'training_called':False,'reserved_test_rows_read':0}
    for use in ('contrast_hypothesis','preservation_control'):
        summary['conditions'][use]={}
        for c in CONDITIONS:
            group=[r for r in metrics if r['condition']==c and r['use']==use]
            summary['conditions'][use][c]={'expected':sum(j['condition']==c and j['use']==use for j in key['jobs']),
                'accepted':len(group),'near_copies':sum(r['source_similarity']>=.9 for r in group),
                'exact_copies':sum(r['exact_source_copy'] for r in group),
                'mean_source_similarity_accepted_only':statistics.mean(r['source_similarity'] for r in group) if group else None}
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'package_manifest_sha256':file_sha(package/'manifest.json'),'code_sha256':file_sha(Path(__file__)),
        'original_response_hashes':coverage['response_file_hashes'],'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('CPU import finished; inspect coverage. Semantic/style quality remains unassessed; no models or GPU accessed.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--check-package',action='store_true');a.add_argument('--audit-responses',action='store_true')
    p.add_argument('--package',type=Path);p.add_argument('--responses-dir',type=Path);p.add_argument('--output-dir',type=Path);args=p.parse_args(argv)
    if args.prepare:
        if args.output_dir is None:p.error('--output-dir required')
        prepare(args.output_dir)
    elif args.check_package:
        if args.package is None:p.error('--package required')
        check_package(args.package);print('Ten exact frozen prompts verified. No generation executed.')
    else:
        if args.package is None or args.responses_dir is None or args.output_dir is None:p.error('--package, --responses-dir and --output-dir required')
        audit_responses(args.package,args.responses_dir,args.output_dir)


if __name__=='__main__':main()
