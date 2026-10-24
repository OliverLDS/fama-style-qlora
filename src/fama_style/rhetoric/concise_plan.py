"""Eight-generation, training-only oracle-plan realization diagnostic. No trainer."""
import argparse
import collections
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import re
import statistics
import time

from fama_style.prompting.corpus import make_prompt
from fama_style.research.claims import sentences,sha,risk_flags,tokens
from fama_style.research.signal import fit_profile,distance,overlap
from .extract import extract
from .experiment import secret_token,with_heartbeat,write_json,write_lines,fresh
from .semantics import NLI_MODEL,assess,calibration_controls

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='concise_plan_realization_v1'
TRAIN_SHA='a108416c1935615a7e01bb4d3fa86e0818b961bfb16f15d3883e6905654ee248'
MODEL_ID='google/gemma-3-4b-it'
MODEL_REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
NLI_REVISION='6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
CONDITIONS=('A_ordinary','B_concise_oracle_plan')
WORD_BANDS={'short':(1,16),'medium':(17,32),'long':(33,48),'extended':(49,72)}
SHAPES=('unconstrained','dependent_clause','coordinated_segments')
MAX_NEW_TOKENS=512
CONTEXT_LIMIT=4096


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_lines(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def word_count(text):return len(re.findall(r"\b[\w’'-]+\b",text))


def word_band(text):
    n=word_count(text)
    for band,(low,high) in WORD_BANDS.items():
        if low<=n<=high:return band
    raise ValueError('Reference sentence outside supported length bands')


def plan_from_annotation(annotation,reference):
    units=annotation['rhetorical_taxonomy']['sentence_structure']
    if ' '.join(' '.join(s['sentence_text'].split()) for s in units)!=' '.join(reference.split()):
        raise ValueError('Reference annotation coverage mismatch')
    if len(sentences(reference))!=len(units):raise ValueError('Sentence counter and annotated reference disagree')
    layouts=[]
    for unit in units:
        clauses=unit['clauses']
        # Positive annotations only: a single spanning clause does not imply simplicity.
        shape='dependent_clause' if any(c['depends_on_clause_ids'] for c in clauses) else (
            'coordinated_segments' if any('coordinat' in c['role']['name'].lower() for c in clauses) else 'unconstrained')
        layouts.append({'word_band':word_band(unit['sentence_text']),'shape_hint':shape})
    plan={'sentence_count':len(units),'sentences':layouts,'max_semicolons':reference.count(';')}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if set(plan)!={'sentence_count','sentences','max_semicolons'}:raise ValueError('Unexpected free-form plan fields')
    if type(plan['sentence_count']) is not int or not 2<=plan['sentence_count']<=6:raise ValueError('Unsupported sentence count')
    if type(plan['max_semicolons']) is not int or not 0<=plan['max_semicolons']<=2:raise ValueError('Unsupported semicolon budget')
    if len(plan['sentences'])!=plan['sentence_count']:raise ValueError('Plan slot count mismatch')
    for slot in plan['sentences']:
        if set(slot)!={'word_band','shape_hint'} or slot['word_band'] not in WORD_BANDS or slot['shape_hint'] not in SHAPES:
            raise ValueError('Unknown or lexicalized plan slot')


def render_plan(plan):
    validate_plan(plan)
    lines=[f'Write exactly {plan["sentence_count"]} complete sentences in one paragraph.',
           f'Use at most {plan["max_semicolons"]} semicolons.']
    for index,slot in enumerate(plan['sentences'],1):
        low,high=WORD_BANDS[slot['word_band']];hint=''
        if slot['shape_hint']=='dependent_clause':hint=' Prefer a dependent construction only where the source already supports that relationship.'
        elif slot['shape_hint']=='coordinated_segments':hint=' Prefer linked grammatical segments only where the source already supplies their relationship.'
        lines.append(f'Sentence {index}: aim for {low}–{high} words.'+hint)
    lines.append('These are presentation controls. Preserve the source claim order, all claims and qualifications. '
                 'Do not add a causal connection, condition, comparison, example or citation to satisfy a pattern. '
                 'Faithfulness takes priority over any length or syntax hint; do not pad the text. Return only the paragraph.')
    return '\n'.join(lines)


def select_four(candidates):
    """Frozen low-cost contrast selection; no model outputs or held-out examples."""
    if any(row['split']!='train' for row in candidates):raise ValueError('Only training candidates allowed')
    choices=[]
    for group in itertools.combinations(candidates,4):
        if len({r['paper_key'] for r in group})!=4:continue
        deltas=[r['plan']['sentence_count']-r['source_sentence_count'] for r in group]
        if sum(d>0 for d in deltas)!=2 or sum(d<0 for d in deltas)!=2:continue
        score=(sum(word_count(r['source'])+word_count(r['reference']) for r in group),
               sum(abs(d) for d in deltas),tuple(sorted(r['id'] for r in group)))
        choices.append((score,group))
    if not choices:raise ValueError('Cannot select four distinct-paper pairs with two count increases and two decreases')
    return sorted(min(choices,key=lambda value:value[0])[1],key=lambda row:row['id'])


def prepare_pack(output):
    fresh(output)
    training=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(training)!=TRAIN_SHA:raise ValueError('Frozen training file changed')
    train=read_lines(training)
    if len(train)!=247 or any(r['split']!='train' for r in train):raise ValueError('Expected frozen 247 training rows')
    base=ROOT/'research_runs/taxonomy_discovery_v1';package=base/'package_v1'
    originals=base/'response_audit_v1/original_annotations.checked.private.jsonl'
    neutral=base/'response_audit_v1/neutral_proposals.checked.private.jsonl'
    references=base/'response_audit_v2/annotations_B.normalized.private.jsonl'
    keypath=package/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json'
    sources=[training,originals,neutral,references,keypath]
    before={str(path.relative_to(ROOT)):file_sha(path) for path in sources}
    for folder,names in ((package,[keypath.name]),(originals.parent,[originals.name,neutral.name]),(references.parent,[references.name])):
        manifest=json.loads((folder/'manifest.json').read_text())
        for name in names:
            if file_sha(folder/name)!=manifest['files'][name]:raise ValueError('Previously audited annotation changed')
    by_id={r['id']:r for r in train};a={r['paragraph_id']:r for r in read_lines(originals)}
    b={r['paragraph_id']:r for r in read_lines(references)};proposals={r['pair_id']:r for r in read_lines(neutral)}
    candidates=[];excluded=collections.Counter()
    for key in json.loads(keypath.read_text()):
        proposal=proposals[key['pair_id']];row=by_id[key['training_id']]
        if proposal['decision']!='retain':excluded['unaccepted_rewrite_proposal']+=1;continue
        if proposal['proposed_neutral_text']!=row['plain_english']:raise ValueError('Retained proposal differs from frozen source')
        if word_count(row['plain_english'])>140 or word_count(row['completion'])>160:excluded['length_cap']+=1;continue
        source_units=b[key['reference_id']]['rhetorical_taxonomy']['sentence_structure']
        if len(source_units)!=len(sentences(row['plain_english'])):excluded['source_sentence_counter_disagreement']+=1;continue
        try:plan=plan_from_annotation(a[key['original_item_id']],row['completion'])
        except ValueError:excluded['unsupported_reference_plan']+=1;continue
        if plan['sentence_count']==len(source_units):excluded['no_sentence_count_contrast']+=1;continue
        candidates.append({'id':row['id'],'paper_key':row['paper_key'],'split':'train','source':row['plain_english'],
            'reference':row['completion'],'source_sentence_count':len(source_units),'plan':plan,
            'reference_annotation_sha256':sha(json.dumps(a[key['original_item_id']],sort_keys=True,ensure_ascii=False)),
            'annotation_uncertainty':a[key['original_item_id']]['review_notes'],
            'neutral_reference_review':'retained_by_ChatGPT_not_independently_verified'})
    selected=select_four(candidates)
    pack={'protocol':PROTOCOL,'parent_training_sha256':TRAIN_SHA,'rows':selected,'style_profile':fit_profile(train),
        'selection':{'eligible_candidates':len(candidates),'exclusion_counts':dict(excluded),
            'rule':'Four distinct training papers; two sentence-count increases, two decreases; minimize source+reference word budget; deterministic ties',
            'selection_used_model_outputs':False,'held_out_rows_read':0},
        'annotation_input_hashes':before,'scope':'Small training-only oracle diagnostic; no deployment or generalization result; not a test of the entire 132-category taxonomy'}
    path=output/'concise_plan_pack.json';write_json(path,pack)
    for name,h in before.items():
        if file_sha(ROOT/name)!=h:raise ValueError('Source changed during preparation')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':file_sha(path),'inputs_unchanged':True,
        'input_hashes':before,'code_sha256':file_sha(Path(__file__).resolve()),'model_calls':0,'gpu_training_called':False})
    print(json.dumps({'status':'pack_prepared_not_executed','pairs':4,'jobs':8,'pack_sha256':file_sha(path),'output_dir':str(output)}))
    return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha) or file_sha(path)!=expected_sha:raise ValueError('Input pack hash mismatch')
    pack=json.loads(path.read_text())
    if pack['protocol']!=PROTOCOL or pack['parent_training_sha256']!=TRAIN_SHA:raise ValueError('Wrong pack protocol or parent corpus')
    rows=pack['rows']
    if len(rows)!=4 or len({r['id'] for r in rows})!=4 or len({r['paper_key'] for r in rows})!=4:raise ValueError('Wrong pair/paper counts')
    for row in rows:
        if row['split']!='train' or not row['source'].strip() or not row['reference'].strip():raise ValueError('Invalid training row')
        validate_plan(row['plan'])
        if row['source_sentence_count']!=len(sentences(row['source'])):raise ValueError('Source sentence count changed')
        if len(sentences(row['reference']))!=row['plan']['sentence_count']:raise ValueError('Reference/plan count mismatch')
        if [word_band(part) for part in sentences(row['reference'])]!=[slot['word_band'] for slot in row['plan']['sentences']]:
            raise ValueError('Reference/plan length bands changed')
    deltas=[r['plan']['sentence_count']-r['source_sentence_count'] for r in rows]
    if sum(d>0 for d in deltas)!=2 or sum(d<0 for d in deltas)!=2:raise ValueError('Count-contrast balance changed')
    return pack


def jobs_for(pack):
    result=[]
    for index,row in enumerate(pack['rows']):
        order=CONDITIONS if index%2==0 else CONDITIONS[::-1]
        for condition in order:
            prompt=make_prompt(row['source'])
            if condition==CONDITIONS[1]:prompt+='\n\nPresentation plan:\n'+render_plan(row['plan'])
            result.append({'id':row['id'],'condition':condition,'prompt':prompt,'prompt_sha256':sha(prompt),'source_sha256':sha(row['source'])})
    return result


def plan_metrics(text,plan):
    parts=sentences(text);count_match=len(parts)==plan['sentence_count'];lengths=[];shapes=[]
    for part,slot in zip(parts,plan['sentences']):
        low,high=WORD_BANDS[slot['word_band']];lengths.append(low<=word_count(part)<=high)
        if slot['shape_hint']!='unconstrained':
            syntax=extract(part)['structure']['sentences'][0]['syntax']
            shapes.append(syntax in (('subordinated','mixed') if slot['shape_hint']=='dependent_clause' else ('coordinated','mixed')))
    # Missing/extra sentences never receive a perfect length score from a partial zip.
    return {'sentence_count':len(parts),'desired_sentence_count':plan['sentence_count'],'sentence_count_match':count_match,
        'sentence_count_absolute_error':abs(len(parts)-plan['sentence_count']),
        'word_band_match':sum(lengths)/max(len(parts),plan['sentence_count']),
        'semicolon_budget_match':text.count(';')<=plan['max_semicolons'],
        'shape_hint_surface_match':statistics.mean(shapes) if count_match and shapes else None,
        'shape_measurement':'Limited surface syntax proxy, not verified parse or preserved semantic relation'}


def evaluate(pack,jobs,outputs,semantic=None,pair_checks=None,calibration=None,synthetic=False):
    expected={(job['id'],job['condition']):job for job in jobs};seen=set();records=[]
    if len(expected)!=len(jobs) or len(outputs)!=len(expected):raise ValueError('Duplicate jobs or incomplete generation set')
    by_id={r['id']:r for r in pack['rows']}
    for output in outputs:
        key=(output['id'],output['condition']);job=expected.get(key)
        if key in seen or job is None or output['prompt_sha256']!=job['prompt_sha256']:raise ValueError('Duplicate/swapped generation')
        if not output.get('output','').strip() or output.get('hit_output_limit') is not False or output.get('synthetic') is not synthetic:
            raise ValueError('Empty, truncated or misreported generation')
        seen.add(key);row=by_id[output['id']];text=output['output'];nli=(semantic or {}).get(key)
        records.append({'id':row['id'],'condition':output['condition'],**plan_metrics(text,row['plan']),
            'source_similarity':overlap(row['source'],text),'near_copy':overlap(row['source'],text)>=.90,
            'exact_copy':' '.join(row['source'].split())==' '.join(text.split()),
            'reference_similarity_diagnostic_only':overlap(row['reference'],text),
            'semantic_surface_flags':risk_flags(row['source'],text),'nli':nli,
            'nli_status':nli['status'] if nli else 'not_run',
            'style_gain_proxy':distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']),
            'words_per_sentence':word_count(text)/max(1,len(sentences(text))),
            'input_tokens':output.get('input_tokens'),'output_tokens':output.get('output_tokens'),
            'generation_seconds':output.get('generation_seconds')})
    aggregate={}
    for condition in CONDITIONS:
        group=[r for r in records if r['condition']==condition]
        aggregate[condition]={'rows':len(group),**{key:statistics.mean(r[key] for r in group) for key in
            ('sentence_count_match','sentence_count_absolute_error','word_band_match','semicolon_budget_match','source_similarity','near_copy','exact_copy','style_gain_proxy','words_per_sentence')},
            'nli_proxy_passes':sum(r['nli_status']=='proxy_consistent' for r in group) if calibration and calibration['passed'] else None,
            'nli_unknown':sum(r['nli_status']=='unknown_over_context' for r in group),
            'generation_seconds':sum(r['generation_seconds'] or 0 for r in group),
            'input_tokens':sum(r['input_tokens'] or 0 for r in group),'output_tokens':sum(r['output_tokens'] or 0 for r in group)}
    lookup={(r['id'],r['condition']):r for r in records};paired=[]
    for row in pack['rows']:
        a,b=(lookup[(row['id'],condition)] for condition in CONDITIONS)
        paired.append({'id':row['id'],'count_error_reduction':a['sentence_count_absolute_error']-b['sentence_count_absolute_error'],
            'style_gain_difference':b['style_gain_proxy']-a['style_gain_proxy'],
            'ordinary_nli':a['nli_status'],'planned_nli':b['nli_status']})
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference',
        'conditions':aggregate,'paired_differences':paired,'pair_source_reference_nli':pair_checks,
        'nli_controls':calibration,'training_called':False,'held_out_rows_used':0,'oracle_is_deployable':False,
        'semantic_accuracy':None,'full_mapping_training_ready':False,
        'limitations':'Four training pairs; joint count/length/clause/semicolon controls; heuristic adherence/style proxies; NLI is not proof. Pair-fidelity flags are reported without post-hoc sample removal.'}


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256']:raise ValueError('Changed prompt')
    if not ids or len(ids)+MAX_NEW_TOKENS>CONTEXT_LIMIT:raise ValueError('Context overflow; no truncation allowed')
    return {'id':job['id'],'condition':job['condition'],'input_tokens':len(ids),
        'prompt_sha256':job['prompt_sha256'],'input_ids_sha256':sha(json.dumps(ids))}


def run_training(pack,output,uploaded_pack_sha):
    """Inference only; lazy ML imports follow the repository entry-point convention."""
    fresh(output);jobs=jobs_for(pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'uploaded_pack_identity.json',{'sha256':uploaded_pack_sha,'format':'uploaded file bytes'})
    status={'complete':False,'protocol':PROTOCOL,'training_called':False,'held_out_rows_used':0,'completed_jobs':0,'total_jobs':len(jobs)}
    started=time.monotonic();outputs=[]
    def phase(name,**extra):
        status.update(phase=name,**extra);write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        phase('loading_inference_runtime')
        import os
        import gc
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import (AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,
            Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList)
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one Kaggle GPU before importing torch')
        torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats(0)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select Kaggle T4 x2 for the pinned diagnostic')
        phase('checking_secret');token=secret_token()
        phase('loading_pinned_processor')
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        def tokenize(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],
                tokenize=True,add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[]
        for job in jobs:audits.append(audit_tokens(job,tokenize(job)['input_ids'][0].tolist()))
        write_json(output/'token_audit.json',audits)
        phase('loading_pinned_gemma')
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
            token=token,dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):
            raise RuntimeError('Unexpected model placement')
        model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('Nonfinite generation scores')
                return scores
        by_key={(r['id'],r['condition']):r for r in audits}
        for job in jobs:
            phase('generation',completed_jobs=len(outputs))
            batch=tokenize(job);ids=batch['input_ids'][0].tolist();audit=audit_tokens(job,ids)
            if audit!=by_key[(job['id'],job['condition'])]:raise ValueError('Tokenized prompt changed')
            batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            result={'id':job['id'],'condition':job['condition'],'prompt_sha256':job['prompt_sha256'],'output':text,
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before,
                'hit_output_limit':bool(new.numel()>=MAX_NEW_TOKENS and int(new[-1]) not in eos),'synthetic':False}
            outputs.append(result);write_lines(output/'generations.private.jsonl',outputs)
            if not text or result['hit_output_limit']:raise RuntimeError('Empty/cutoff output; partial results remain incomplete')
            phase('generation',completed_jobs=len(outputs))
        gemma_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        phase('loading_independent_nli_on_gpu');nli_started=time.monotonic()
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer load')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,
            token=False).to('cuda:0'),'NLI model load')
        nli.eval();nli.requires_grad_(False)
        labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Unexpected NLI labels')
        limit=min(512,int(getattr(nli.config,'max_position_embeddings',512)),int(tokenizer.model_max_length))
        cache={};cache_hits=0
        def predict(premise,hypothesis):
            nonlocal cache_hits
            key=(sha(premise),sha(hypothesis))
            if key in cache:cache_hits+=1;return cache[key]
            batch=tokenizer(premise,hypothesis,truncation=False,return_tensors='pt')
            if int(batch['input_ids'].shape[-1])>limit:cache[key]=None;return None
            with torch.inference_mode():
                probs=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
            cache[key]={labels[i]:float(p) for i,p in enumerate(probs)}
            return cache[key]
        calibration=with_heartbeat(lambda:calibration_controls(predict),'NLI controls');write_json(output/'nli_controls.json',calibration)
        semantic={};pair_checks={}
        for index,row in enumerate(pack['rows'],1):
            phase('pair_fidelity_assessment',completed_pairs=index-1)
            pair_checks[row['id']]=assess(row['source'],row['reference'],predict)
        source_by_id={row['id']:row['source'] for row in pack['rows']}
        for index,result in enumerate(outputs,1):
            phase('output_semantic_assessment',completed_nli_outputs=index-1)
            semantic[(result['id'],result['condition'])]=with_heartbeat(lambda:assess(source_by_id[result['id']],result['output'],predict),'NLI assessment')
        records,summary=evaluate(pack,jobs,outputs,semantic,pair_checks,calibration)
        write_lines(output/'metrics.private.jsonl',records)
        summary.update(model_id=MODEL_ID,model_revision=MODEL_REVISION,nli_model=NLI_MODEL,nli_revision=NLI_REVISION,
            nli_device='cuda:0 after Gemma unload',nli_seconds=time.monotonic()-nli_started,total_seconds=time.monotonic()-started,
            nli_cache_entries=len(cache),nli_cache_hits=cache_hits,gemma_memory=gemma_memory,
            nli_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)},
            generation={'do_sample':False,'seed':42,'max_new_tokens':MAX_NEW_TOKENS,'context_limit':CONTEXT_LIMIT,'truncate':False},
            versions={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')})
        write_json(output/'evaluation.json',summary)
        phase('complete',complete=True,completed_jobs=len(outputs),completed_nli_outputs=len(outputs),completed_pairs=len(pack['rows']))
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'training_called':False,'input_pack_sha256':uploaded_pack_sha,
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as error:
        failed_at=status.get('phase');locations=[];tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:locations.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,failed_at=failed_at,error_code='concise_plan_inference_failed',local_line_numbers=locations)
        raise RuntimeError('Concise-plan inference failed; inspect run_status.json and saved partial files. No training was called.') from None


def synthetic_pack():
    examples=[('Firms report revenue and investors read it.','Firms report revenue. Investors read it.'),
        ('Firms report sales, investors read reports, and analysts examine costs.','Firms report sales. Investors read reports. Analysts examine costs.'),
        ('Firms publish reports. Investors read them. Analysts examine costs.','Firms publish reports; investors read them. Analysts examine costs.'),
        ('Firms report revenue. Investors read reports. Analysts compare costs. Managers plan investment.',
         'Firms report revenue; investors read reports. Analysts compare costs. Managers plan investment.')]
    rows=[];profile_rows=[]
    for index,(source,reference) in enumerate(examples):
        plan={'sentence_count':len(sentences(reference)),
            'sentences':[{'word_band':word_band(part),'shape_hint':'unconstrained'} for part in sentences(reference)],'max_semicolons':reference.count(';')}
        rows.append({'id':f'synthetic-{index}','paper_key':f'SYN-{index}','split':'train','source':source,'reference':reference,
                     'source_sentence_count':len(sentences(source)),'plan':plan})
        profile_rows.append({'split':'train','plain_english':source,'completion':reference})
    return {'protocol':PROTOCOL,'parent_training_sha256':'synthetic_not_frozen_corpus','rows':rows,'style_profile':fit_profile(profile_rows)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare-pack',action='store_true');actions.add_argument('--dry-run',action='store_true')
    actions.add_argument('--execute-inference',action='store_true')
    parser.add_argument('--pack',type=Path);parser.add_argument('--expected-pack-sha');parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.dry_run:
        fresh(args.output_dir);pack=synthetic_pack();jobs=jobs_for(pack);sources={r['id']:r['source'] for r in pack['rows']}
        outputs=[{'id':job['id'],'condition':job['condition'],'prompt_sha256':job['prompt_sha256'],
                  'output':sources[job['id']],'hit_output_limit':False,'synthetic':True} for job in jobs]
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        write_json(args.output_dir/'synthetic_evaluation.json',summary);write_lines(args.output_dir/'synthetic_jobs.jsonl',jobs)
        print(json.dumps({'status':'synthetic_wiring_check_complete','jobs':len(jobs),'model_calls':0,'training_called':False}));return
    if args.pack is None or args.expected_pack_sha is None:parser.error('--pack and --expected-pack-sha are required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:
        run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':args.expected_pack_sha,'pairs':4,'jobs':8,'training_called':False})
        print('Preflight passed; no Secret access, model download or inference performed.')


if __name__=='__main__':main()
