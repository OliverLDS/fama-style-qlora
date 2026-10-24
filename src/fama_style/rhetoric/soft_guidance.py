"""CPU preparation of a gated 21-example soft-plan comparison; no GPU runner."""
import argparse
import json
from pathlib import Path

from fama_style.prompting.corpus import make_prompt
from fama_style.research.claims import sentences,sha
from fama_style.research.signal import fit_profile
from .claim_constrained import constraint_ledger
from .concise_plan import TRAIN_SHA,file_sha,read_lines
from .experiment import fresh,write_json,write_lines
from .extract import extract

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='soft_guidance_validation_v1'
VALIDATION_SHA='f1c75c29fa93f8458e7a3b2f8996bbd13ff023f92e4a74f1d086d22af5e546b8'
CONDITIONS=('A_claims_only','B_claims_soft_plan')


def soft_plan(source,profile):
    if profile.get('fit_split')!='train':raise ValueError('Only training-fitted priors allowed')
    f=profile['features']['words_per_sentence']
    merge=(f['mean_paired_delta']>.5 and f['positive_pair_fraction']>=.75 and len(sentences(source))>1)
    structure=extract(source)['structure']
    return {'paragraph_structure':{'operation':'retain_source_argument_order',
                'observed_source_organization':structure['paragraph']['organization'],
                'source_unit_order':['c'+str(i) for i in range(1,len(sentences(source))+1)]},
            'sentence_structure':{'operation':'optional_adjacent_grouping' if merge else 'retain_or_rephrase',
                'hard_sentence_count':None,'force_grouping':False,
                'evidence':'training_reference_sentence_length_direction' if merge else 'no_eligible_count_free_style_operation'},
            'language_style':{'operation':'no_learned_lexical_prescription',
                'reason':'Frozen surface profile does not establish safe distinctive lexical transformations'},
            'introduces_semantic_relation':False}


def jobs_for(pack):
    rows=pack['rows']
    if len(rows)!=21 or len({r['id'] for r in rows})!=21 or any(r['split']!='validation' for r in rows):raise ValueError('Expected 21 unique development-validation rows')
    jobs=[]
    for index,row in enumerate(rows):
        base=make_prompt(row['source'])+'\n\nSource-only preservation ledger:\n'+json.dumps(constraint_ledger(row['source']),ensure_ascii=False,separators=(',',':'))
        base+='\nPreserve all source claims, actors, quantities and their assignments, negation, bounds, uncertainty, conditions, qualifications, attribution and causal direction. The ledger is a fallible aid; the complete source is authoritative. Do not invent explanations or conclusions. Choose the sentence count freely. Return only the paragraph.'
        for condition in CONDITIONS if index%2==0 else CONDITIONS[::-1]:
            prompt=base
            if condition==CONDITIONS[1]:
                prompt+='\n\nOptional source-bound presentation plan:\n'+json.dumps(soft_plan(row['source'],pack['style_profile']),separators=(',',':'))
                prompt+='\nYou may group adjacent statements into a more integrated sentence only where the source already supports their relationship. Do not add a causal link or condition. Keep qualifications attached to their claims. Do not merge, pad or formalize wording merely to satisfy a target pattern. The plan is optional; faithfulness has priority.'
            jobs.append({'id':row['id'],'split':'validation','condition':condition,'prompt':prompt,
                'source_sha256':sha(row['source']),'prompt_sha256':sha(prompt),'oracle':False})
    return jobs


def prepare(training,validation,benchmark_summary,output):
    if file_sha(training)!=TRAIN_SHA or file_sha(validation)!=VALIDATION_SHA:raise ValueError('Frozen train/validation bytes changed')
    train,val=read_lines(training),read_lines(validation)
    if len(train)!=247 or any(r['split']!='train' for r in train):raise ValueError('Wrong training split')
    if len(val)!=21 or any(r['split']!='validation' for r in val):raise ValueError('Wrong validation split')
    if {r['id'] for r in train}&{r['id'] for r in val} or {r['paper_key'] for r in train}&{r['paper_key'] for r in val}:raise ValueError('Train/validation overlap')
    benchmark=json.loads(benchmark_summary.read_text());manifest=json.loads((benchmark_summary.parent/'manifest.json').read_text())
    if benchmark.get('protocol')!='claim_checker_benchmark_v1' or manifest['files']['summary.json']!=file_sha(benchmark_summary):raise ValueError('Unverified benchmark result')
    ready=benchmark['readiness_gate']['passed']
    pack={'protocol':PROTOCOL,'parent_training_sha256':TRAIN_SHA,'validation_sha256':VALIDATION_SHA,
        'style_profile':fit_profile(train),
        'rows':[{'id':r['id'],'paper_key':r['paper_key'],'split':'validation','source':r['plain_english'],'reference':r['completion']} for r in val],
        'execution_gate':{'ready':ready,'reason':'Frozen checker passed synthetic gate' if ready else 'Frozen checker failed synthetic gate; do not generate more comparison text yet',
            'benchmark_summary_sha256':file_sha(benchmark_summary)},
        'scope':'Non-oracle fixed soft prior, not a learned taxonomy mapper. Existing 21 development examples, not fresh validation. No reserved test file read.'}
    jobs=jobs_for(pack);fresh(output);write_json(output/'soft_guidance_pack.private.json',pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':21,'jobs':42,'execution_ready':ready,
        'training_called':False,'model_calls':0,'reserved_test_rows_used':0,'code_sha256':file_sha(Path(__file__)),
        'input_hashes':{str(p):file_sha(p) for p in (training,validation,benchmark_summary)},
        'files':{n:file_sha(output/n) for n in ('soft_guidance_pack.private.json','jobs.private.jsonl')}})
    print(json.dumps({'pairs':21,'jobs':42,'execution_ready':ready,'model_calls':0,'reserved_test_rows_used':0}));return pack


def execution_allowed(pack):
    """A passing synthetic gate is necessary, not proof of domain reliability."""
    if pack['execution_gate']['ready'] is not True:raise ValueError('Claim-checker readiness gate failed; comparison execution remains blocked')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=ROOT/'data/fama_corpus_v1/train.jsonl')
    p.add_argument('--validation',type=Path,default=ROOT/'data/fama_corpus_v1/validation.jsonl')
    p.add_argument('--benchmark-summary',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);prepare(a.training,a.validation,a.benchmark_summary,a.output_dir)


if __name__=='__main__':main()
