"""Bounded matched direct/plan-conditioned realizer pilot; CPU preparation by default."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time

from fama_style.evaluation import corpus_baseline as baseline
from fama_style.training import corpus_smoke as smoke
from fama_style.rhetoric import bound_grouping as grouping
from fama_style.rhetoric.concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from fama_style.rhetoric.experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research.claims import sha,risk_flags
from fama_style.research.signal import fit_profile,features,overlap,distance
from fama_style.research import claim_verifier_v3 as verifier

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='plan_realizer_pilot_v1'
PLAN_VERSION='realizer_surface_plan_v1'
VALIDATION_SHA='f1c75c29fa93f8458e7a3b2f8996bbd13ff023f92e4a74f1d086d22af5e546b8'
ARMS=('direct','planned')
CONFIG={'epochs':4,'learning_rate':5e-5,'rank':8,'alpha':16,'dropout':0.05,
    'accumulation':2,'seed':42,'max_length':1536,'context_limit':4096,'max_new_tokens':512}
BANDS={'short':(1,16),'medium':(17,32),'long':(33,48),'extended':(49,72)}
COMMON=(grouping.CONCRETE_STYLE+
    'Rewrite the source as one formal economics paragraph. Preserve its actors, substantive claims, '
    'quantities and their roles, conditions, uncertainty, negation and causal direction. '
    'Do not invent mechanisms, calculations or implications. Prioritize substantive fidelity over a presentation hint. '
    'Minor phrasing changes are acceptable; return only the paragraph.\n\n')


def presentation_plan(text):
    parsed=grouping.output_units(text)
    if parsed['ambiguous'] or not 2<=len(parsed['units'])<=6:raise ValueError('Unsupported or uncertain sentence partition')
    bands=[]
    for sentence in parsed['units']:
        count=len(re.findall(r"\b[\w’'-]+\b",sentence))
        band=next((name for name,(lo,hi) in BANDS.items() if lo<=count<=hi),None)
        if band is None:raise ValueError('Unsupported sentence word band')
        bands.append(band)
    if text.count(';')>2:raise ValueError('Unsupported semicolon budget')
    return {'version':PLAN_VERSION,'paragraph':{'sentence_count':len(bands)},
        'sentences':[{'word_band':b} for b in bands],
        'language':{'semicolon_budget':text.count(';')}}


def validate_plan(plan):
    if set(plan)!={'version','paragraph','sentences','language'} or plan['version']!=PLAN_VERSION:raise ValueError('Invalid plan schema')
    if set(plan['paragraph'])!={'sentence_count'} or type(plan['paragraph']['sentence_count']) is not int or not 2<=plan['paragraph']['sentence_count']<=6:raise ValueError('Invalid count')
    if len(plan['sentences'])!=plan['paragraph']['sentence_count']:raise ValueError('Plan slots mismatch')
    if any(set(s)!={'word_band'} or s['word_band'] not in BANDS for s in plan['sentences']):raise ValueError('Lexicalized or invalid slot')
    if set(plan['language'])!={'semicolon_budget'} or type(plan['language']['semicolon_budget']) is not int or not 0<=plan['language']['semicolon_budget']<=2:raise ValueError('Invalid language controls')


def prompt_for(row,arm):
    if arm not in ARMS:raise ValueError('Unknown prompt arm')
    prompt=COMMON
    if arm=='planned':
        validate_plan(row['plan'])
        prompt+=('Presentation controls (oracle structural side information, not additional claims):\n'+
            json.dumps(row['plan'],sort_keys=True)+'\nWord bands: short 1-16, medium 17-32, long 33-48, extended 49-72 words. '
            'Aim for the sentence count, word bands and semicolon ceiling only when the source permits. '
            'Retain claim order; a sentence boundary does not authorize a new semantic relationship.\n\n')
    return prompt+'Source:\n'+row['source']


def quantity_tokens(text):
    # Numeric spellings/ranges only; not an economic role or equivalence parser.
    return re.findall(r'\d+(?:\.\d+)?',text)


def pair_audit(row):
    return {'id':row['id'],'source_sha256':sha(row['source']),'reference_sha256':sha(row['reference']),
        'source_numbers':quantity_tokens(row['source']),'reference_numbers':quantity_tokens(row['reference']),
        'number_multiset_matches':Counter(quantity_tokens(row['source']))==Counter(quantity_tokens(row['reference'])),
        'surface_flags':risk_flags(row['source'],row['reference']),
        'source_target_overlap':overlap(row['source'],row['reference']),
        'semantic_equivalence_verified':False,'numeric_role_equivalence_verified':False}


def schedules(ids):
    if len(ids)!=12 or len(set(ids))!=12:raise ValueError('Expected 12 unique training rows')
    result=[]
    for epoch in range(CONFIG['epochs']):
        order=list(ids);random.Random(CONFIG['seed']+epoch).shuffle(order)
        result.extend({'epoch':epoch+1,'ids':order[i:i+2]} for i in range(0,len(order),2))
    return result


def runtime_paths():
    paths=list((ROOT/'src/fama_style').rglob('*.py'))
    return sorted(paths+[ROOT/grouping.POLICY])


def prepare_pack(output):
    train_path=ROOT/'data/fama_corpus_v1/train.jsonl';val_path=ROOT/'data/fama_corpus_v1/validation.jsonl'
    if file_sha(train_path)!=TRAIN_SHA or file_sha(val_path)!=VALIDATION_SHA:raise ValueError('Frozen split changed')
    train=read_lines(train_path);validation=read_lines(val_path)
    if len(train)!=247 or any(r['split']!='train' for r in train) or len(validation)!=21 or any(r['split']!='validation' for r in validation):raise ValueError('Wrong development split')
    base=ROOT/'research_runs/taxonomy_discovery_v1'
    mapping=base/'package_v1/PRIVATE_MAPPING_DO_NOT_UPLOAD.json'
    prop=base/'response_audit_v1/neutral_proposals.checked.private.jsonl'
    annotations=base/'response_audit_v1/original_annotations.checked.private.jsonl'
    inputs=[train_path,val_path,mapping,prop,annotations]
    for path in (mapping,prop,annotations):
        manifest=json.loads((path.parent/'manifest.json').read_text())
        if file_sha(path)!=manifest['files'][path.name]:raise ValueError('Audited annotation changed')
    proposals={r['pair_id']:r for r in read_lines(prop)};ann={r['paragraph_id']:r for r in read_lines(annotations)};by_id={r['id']:r for r in train}
    selected=[];excluded=[]
    def record(r):
        return {'id':r['id'],'paper_key':r['paper_key'],'split':r['split'],'source':r['plain_english'],
            'reference':r['completion'],'plan':presentation_plan(r['completion']),
            'source_sha256':sha(r['plain_english']),'reference_sha256':sha(r['completion'])}
    for key in json.loads(mapping.read_text()):
        proposal=proposals[key['pair_id']];r=by_id[key['training_id']]
        if proposal['decision']!='retain':continue
        if proposal['proposed_neutral_text']!=r['plain_english']:raise ValueError('Retained source mismatch')
        annotation=ann[key['original_item_id']]
        if ' '.join(' '.join(u['sentence_text'].split()) for u in annotation['rhetorical_taxonomy']['sentence_structure'])!=' '.join(r['completion'].split()):raise ValueError('Target annotation coverage mismatch')
        try:row=record(r)
        except ValueError:excluded.append({'id':r['id'],'reason':'outside_fixed_plan_partition_or_band_limits'});continue
        row['annotation_sha256']=sha(json.dumps(annotation,sort_keys=True,ensure_ascii=False))
        row['prior_neutral_decision']='retain_not_independently_verified'
        row['annotation_sentence_count_agrees']=len(annotation['rhetorical_taxonomy']['sentence_structure'])==row['plan']['paragraph']['sentence_count']
        if not row['annotation_sentence_count_agrees']:raise ValueError('Annotation count disagreement')
        selected.append(row)
    selected=sorted(selected,key=lambda r:r['id'])
    if len(selected)!=12:raise ValueError('Expected 12 fixed retained-plan-compatible training rows')
    candidates=[]
    for r in validation:
        try:candidates.append(record(r))
        except ValueError:continue
    # No generation quality, losses or previous model outputs enter selection.
    dev=sorted(sorted(candidates,key=lambda r:(len(r['source'].split())+len(r['reference'].split()),r['id']))[:4],key=lambda r:r['id'])
    if len(dev)!=4:raise ValueError('Need four plan-compatible development rows')
    # Two known training probes distinguish memorization from development transfer.
    probe=[next(r['id'] for r in selected if r['id']=='FF2000::FF00-DD-0070'),selected[0]['id']]
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'config':CONFIG,
        'train_sha256':TRAIN_SHA,'validation_sha256':VALIDATION_SHA,'train':selected,'development':dev,'training_probe_ids':probe,
        'style_profile':fit_profile(train),'input_hashes':{str(p.relative_to(ROOT)):file_sha(p) for p in inputs},
        'annotation_selection':{'retained_before_limits':13,'selected':12,'excluded':excluded,'development_eligible':len(candidates),'development_selection':'four lowest source+reference word budgets, deterministic ID ties; reference-derived feasibility oracle; no output-based selection'},
        'pair_audits':[pair_audit(r) for r in selected+dev],
        'plan_scope':'Bounded count/word-band/semicolon presentation controls, not the full discovered taxonomy or a learned mapper. Reference-derived plans at development are oracle-only.',
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}}
    fresh(output);write_json(output/'realizer_pack.json',pack)
    write_json(output/'pair_audit.private.json',pack['pair_audits'])
    write_json(output/'manifest.json',{'status':'prepared_not_executed','protocol':PROTOCOL,'pack_sha256':file_sha(output/'realizer_pack.json'),'train_pairs':12,'development_pairs':4,'updates_per_arm':24,'training_called':False,'reserved_test_rows_read':0})
    print('Prepared 12 training and 4 development pairs; two fresh matched adapters, 24 updates each. No training executed.')


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    p=json.loads(path.read_text())
    if p['protocol']!=PROTOCOL or p['config']!=CONFIG or p['model_id']!=MODEL_ID or p['model_revision']!=MODEL_REVISION or p['train_sha256']!=TRAIN_SHA or p['validation_sha256']!=VALIDATION_SHA:raise ValueError('Experiment identity changed')
    if len(p['train'])!=12 or len(p['development'])!=4:raise ValueError('Changed experiment size')
    if len({r['id'] for r in p['train']+p['development']})!=16:raise ValueError('Duplicate or overlapping development/training rows')
    for split,rows in [('train',p['train']),('validation',p['development'])]:
        for r in rows:
            if r['split']!=split or r['source_sha256']!=sha(r['source']) or r['reference_sha256']!=sha(r['reference']):raise ValueError('Changed text or split')
            validate_plan(r['plan'])
            if r['plan']!=presentation_plan(r['reference']):raise ValueError('Reference/plan mismatch')
    if p['pair_audits']!=[pair_audit(r) for r in p['train']+p['development']]:raise ValueError('Changed pair audit')
    if len(set(p['training_probe_ids']))!=2 or not set(p['training_probe_ids'])<={r['id'] for r in p['train']}:raise ValueError('Wrong probes')
    for name,h in p['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    schedules([r['id'] for r in p['train']]);return p


def encode_pair(processor,row,arm):
    old={**row,'prompt':prompt_for(row,arm),'completion':row['reference']}
    expected=baseline.prepare_pair(processor,old)
    # Reuse the proven exact boundary check without its frozen 1024 length cap.
    if expected['full_tokens']>CONFIG['max_length']:raise ValueError('Training context overflow; no truncation')
    full=processor.apply_chat_template([baseline.user_message(old),{'role':'assistant','content':[{'type':'text','text':row['reference']}]}],tokenize=True,add_generation_prompt=False,truncation=False,return_dict=True,return_tensors='pt')
    ids=baseline.input_ids(full);prefix=expected['prompt_tokens']
    if len(ids)!=expected['full_tokens'] or sha(json.dumps(ids[:prefix]))!=expected['prefix_token_ids_sha256']:raise ValueError('Training prefix changed')
    if ' '.join(processor.tokenizer.decode(ids[prefix:],skip_special_tokens=True).split())!=' '.join(row['reference'].split()):raise ValueError('Actual batch target changed')
    labels=[-100]*prefix+ids[prefix:]
    data={name:baseline.input_ids({'input_ids':value}) for name,value in full.items() if name in ('input_ids','attention_mask','token_type_ids')}
    data['labels']=labels
    return data,{**expected,'arm':arm,'shifted_supervised_tokens':sum(v!=-100 for v in labels[1:]),'prompt_masked':True}


def generation_rows(pack):
    train={r['id']:r for r in pack['train']}
    return pack['development']+[train[i] for i in pack['training_probe_ids']]


def output_metrics(pack,row,out,verdict=None):
    text=out['output'];available=bool(text.strip()) and not out['hit_output_limit']
    parsed=grouping.output_units(text) if available else None
    band_hits=0
    if parsed:
        for text_part,slot in zip(parsed['units'],row['plan']['sentences']):
            lo,hi=BANDS[slot['word_band']];band_hits+=lo<=len(re.findall(r"\b[\w’'-]+\b",text_part))<=hi
    return {**out,'available':available,'source_sha256':sha(row['source']),'reference_sha256':sha(row['reference']),
        'output_sha256':sha(text),'source_similarity':overlap(row['source'],text) if available else None,
        'reference_similarity':overlap(row['reference'],text) if available else None,
        'exact_source_copy':' '.join(text.split())==' '.join(row['source'].split()) if available else None,
        'surface_flags':risk_flags(row['source'],text) if available else None,
        'quantity_tokens':quantity_tokens(text),'reference_quantity_multiset_matches':Counter(quantity_tokens(text))==Counter(quantity_tokens(row['reference'])) if available else None,
        'target_sentence_count':row['plan']['paragraph']['sentence_count'],
        'word_band_match':band_hits/max(len(parsed['units']),len(row['plan']['sentences'])) if parsed and not parsed['ambiguous'] else None,
        'semicolon_budget_matches':text.count(';')<=row['plan']['language']['semicolon_budget'] if available else None,
        'count_matches':len(parsed['units'])==row['plan']['paragraph']['sentence_count'] if parsed and not parsed['ambiguous'] else None,
        'style_distance_gain_proxy':distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']) if available else None,
        'features':features(text) if available else None,'verifier':verdict,'claim_drift_severity':'unknown','author_style_accuracy':None}


def validate_outputs(pack,outputs):
    expected={(model,r['id'],arm) for model in ('base',)+ARMS for r in generation_rows(pack) for arm in ARMS}
    seen=set();rows={r['id']:r for r in generation_rows(pack)}
    for out in outputs:
        key=(out['model'],out['id'],out['prompt_arm'])
        if key not in expected or key in seen or out['prompt_sha256']!=sha(prompt_for(rows[out['id']],out['prompt_arm'])) or out['synthetic'] is not False or type(out['hit_output_limit']) is not bool:raise ValueError('Unknown, repeated or changed generation')
        seen.add(key)
    if seen!=expected:raise ValueError('Incomplete generation matrix')



def validate_completion(pack,status,loss_reports,outputs,adapter_states):
    expected=schedules([r['id'] for r in pack['train']])
    if status['completed_arms']!=list(ARMS) or status['training_called'] is not True:raise ValueError('Incomplete arms')
    for arm in ARMS:
        updates=status['updates'][arm]
        if len(updates)!=len(expected):raise ValueError('Incomplete updates')
        for i,(actual,wanted) in enumerate(zip(updates,expected),1):
            if actual['step']!=i or actual['epoch']!=wanted['epoch'] or actual['ids']!=wanted['ids'] or not math.isfinite(actual['loss']) or not math.isfinite(actual['gradient_norm']) or actual['gradient_norm']<=0:raise ValueError('Invalid schedule, loss or gradients')
        if adapter_states['initial'][arm]==adapter_states['trained'][arm]:raise ValueError('Unchanged adapter')
    if adapter_states['initial']['direct']!=adapter_states['initial']['planned'] or adapter_states['both_reload_verified'] is not True:raise ValueError('Initialization or reload mismatch')
    expected_loss={(model,arm) for model in ('base',)+ARMS for arm in ARMS};seen=set()
    for report in loss_reports:
        key=(report['model'],report['prompt_arm'])
        if key not in expected_loss or key in seen:raise ValueError('Unknown or duplicate loss report')
        seen.add(key)
        if [r['id'] for r in report['rows']]!=[r['id'] for r in pack['development']]:raise ValueError('Wrong loss split')
        if any(not math.isfinite(r['loss']) or r['loss']<0 or r['supervised_tokens']<=0 for r in report['rows']):raise ValueError('Invalid loss rows')
        average=sum(r['loss']*r['supervised_tokens'] for r in report['rows'])/sum(r['supervised_tokens'] for r in report['rows'])
        if not math.isclose(report['loss'],average,abs_tol=1e-10,rel_tol=1e-10):raise ValueError('Wrong weighted loss')
    if seen!=expected_loss:raise ValueError('Missing loss reports')
    validate_outputs(pack,outputs)


def run_training(pack,output,pack_sha):
    """Explicitly authorized GPU path; all ML imports remain inside this function."""
    fresh(output);started=time.monotonic();outputs=[];loss_reports=[]
    status={'protocol':PROTOCOL,'complete':False,'training_called':False,'reserved_test_rows_read':0,
        'pack_sha256':pack_sha,'planned_updates_per_arm':24,'completed_arms':[],'updates':{a:[] for a in ARMS}}
    def phase(name):
        status['phase']=name;write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        phase('loading_runtime')
        import torch
        from transformers import (AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,
            Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList)
        from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict,prepare_model_for_kbit_training
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one T4')
        baseline.initialize_cuda(torch)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select Kaggle T4')
        token=secret_token()
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        all_rows=pack['train']+pack['development'];by_id={r['id']:r for r in all_rows};prepared={};audits=[]
        for row in all_rows:
            for arm in ARMS:
                data,audit=encode_pair(processor,row,arm);prepared[(row['id'],arm)]=data;audits.append(audit)
        write_json(output/'label_audit.json',{'records':audits,'truncated':False})
        def tokenize(row,arm):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':prompt_for(row,arm)}]}],tokenize=True,add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        generation_audits=[]
        for row in generation_rows(pack):
            for arm in ARMS:
                ids=tokenize(row,arm)['input_ids'][0].tolist()
                if not ids or len(ids)+CONFIG['max_new_tokens']>CONFIG['context_limit']:raise ValueError('Generation context overflow')
                generation_audits.append({'id':row['id'],'prompt_arm':arm,'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))})
        write_json(output/'generation_token_audit.json',generation_audits)
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token,
            dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        baseline.validate_model_placement(model)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('Nonfinite generation scores')
                return scores
        def batch(key,arm):
            return {k:torch.tensor([v],dtype=torch.long,device='cuda:0') for k,v in prepared[(key,arm)].items()}
        def evaluate_loss(model_label):
            model.eval()
            for arm in ARMS:
                records=[]
                with torch.inference_mode():
                    for row in pack['development']:
                        data=prepared[(row['id'],arm)];loss=float(model(**batch(row['id'],arm),use_cache=False).loss.item())
                        if not math.isfinite(loss):raise RuntimeError('Nonfinite development loss')
                        records.append({'id':row['id'],'loss':loss,'supervised_tokens':sum(v!=-100 for v in data['labels'][1:])})
                loss_reports.append({'model':model_label,'prompt_arm':arm,'rows':records,'loss':sum(r['loss']*r['supervised_tokens'] for r in records)/sum(r['supervised_tokens'] for r in records)})
                write_json(output/'development_losses.json',loss_reports)
        def generate(model_label):
            model.eval()
            for row in generation_rows(pack):
                for arm in ARMS:
                    phase('generation_'+model_label+'_'+str(len(outputs)+1))
                    inputs=tokenize(row,arm);ids=inputs['input_ids'][0].tolist()
                    audit=next(a for a in generation_audits if a['id']==row['id'] and a['prompt_arm']==arm)
                    if audit['input_ids_sha256']!=sha(json.dumps(ids)):raise ValueError('Changed generation tokens')
                    before=time.monotonic()
                    with torch.inference_mode():
                        result=with_heartbeat(lambda:model.generate(**{k:v.to('cuda:0') for k,v in inputs.items()},do_sample=False,
                            max_new_tokens=CONFIG['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,
                            remove_invalid_values=False,logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
                    new=result[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                    text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
                    outputs.append({'model':model_label,'id':row['id'],'split':row['split'],'prompt_arm':arm,'prompt_sha256':sha(prompt_for(row,arm)),
                        'output':text,'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before,
                        'hit_output_limit':bool(new.numel()>=CONFIG['max_new_tokens'] and int(new[-1]) not in eos),'synthetic':False})
                    write_lines(output/'generations.private.jsonl',outputs)
        phase('base_loss_and_generation');evaluate_loss('base');generate('base')
        bare_model=model
        initial_states={};trained_states={}
        for arm in ARMS:
            phase('prepare_'+arm);torch.manual_seed(CONFIG['seed'])
            bare_model.config.use_cache=False;bare_model.config.text_config.use_cache=False
            model=prepare_model_for_kbit_training(bare_model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
            matched=[name for name,_ in model.named_modules() if re.fullmatch(smoke.TARGET_PATTERN,name)]
            if len(matched)!=2*model.config.text_config.num_hidden_layers:raise RuntimeError('Changed adapted module count')
            model=get_peft_model(model,LoraConfig(r=CONFIG['rank'],lora_alpha=CONFIG['alpha'],lora_dropout=CONFIG['dropout'],target_modules=smoke.TARGET_PATTERN,bias='none',task_type='CAUSAL_LM',revision=MODEL_REVISION))
            parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
            if not parameters or any('lora_' not in n or 'language_model' not in n for n in parameters):raise RuntimeError('Only language LoRA may train')
            def adapter_hashes():
                return {n:hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for n,v in get_peft_model_state_dict(model,save_embedding_layers=False).items()}
            initial_states[arm]=adapter_hashes()
            if arm=='planned' and initial_states[arm]!=initial_states['direct']:raise RuntimeError('Adapters did not share initialization')
            optimizer=torch.optim.AdamW(parameters.values(),lr=CONFIG['learning_rate'],weight_decay=0.0)
            status['training_called']=True
            for index,update in enumerate(schedules([r['id'] for r in pack['train']]),1):
                phase('train_'+arm+'_'+str(index)+'_of_24');model.train();optimizer.zero_grad(set_to_none=True);values=[];update_started=time.monotonic()
                for key in update['ids']:
                    def step():
                        loss=model(**batch(key,arm),use_cache=False).loss
                        if not bool(torch.isfinite(loss).all()):raise RuntimeError('Nonfinite training loss')
                        value=float(loss.detach().item());(loss/len(update['ids'])).backward();return value
                    values.append(with_heartbeat(step,'Training microbatch'))
                gradients=[p.grad for p in parameters.values() if p.grad is not None]
                if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):raise RuntimeError('Invalid adapter gradients')
                norm=float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.0,error_if_nonfinite=True))
                if not math.isfinite(norm) or norm<=0:raise RuntimeError('Invalid gradient norm')
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in parameters.values()):raise RuntimeError('Invalid adapter weights')
                status['updates'][arm].append({'step':index,**update,'loss':sum(values)/len(values),'gradient_norm':norm,'training_seconds':time.monotonic()-update_started,
                    'input_tokens':sum(len(prepared[(key,arm)]['input_ids']) for key in update['ids']),
                    'supervised_tokens':sum(sum(v!=-100 for v in prepared[(key,arm)]['labels'][1:]) for key in update['ids'])})
                write_json(output/'training_log.json',status['updates'])
            optimizer.zero_grad(set_to_none=True);trained_states[arm]=adapter_hashes()
            if trained_states[arm]==initial_states[arm]:raise RuntimeError('Adapter unchanged')
            phase('save_and_reload_'+arm)
            directory=output/('adapter_'+arm);model.save_pretrained(directory,safe_serialization=True,save_embedding_layers=False)
            del optimizer,parameters,gradients
            bare_model=model.unload();del model
            if any('lora_' in n for n,_ in bare_model.named_parameters()):raise RuntimeError('Adapter unload failed')
            model=PeftModel.from_pretrained(bare_model,directory,is_trainable=False)
            if adapter_hashes()!=trained_states[arm]:raise RuntimeError('Reloaded adapter changed')
            baseline.validate_model_placement(model);evaluate_loss(arm);generate(arm)
            bare_model=model.unload();del model;gc.collect();torch.cuda.empty_cache()
            status['completed_arms'].append(arm)
        adapter_states={'initial':initial_states,'trained':trained_states,'both_reload_verified':True}
        validate_completion(pack,status,loss_reports,outputs,adapter_states)
        memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        write_json(output/'adapter_hashes.json',adapter_states)
        del bare_model,processor,prepared;gc.collect();torch.cuda.empty_cache()
        phase('loading_independent_diagnostics')
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False,dtype=torch.float32).to('cuda:0'),'NLI model')
        nli.eval();nli.requires_grad_(False);labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Unexpected NLI labels')
        limit=min(512,int(tokenizer.model_max_length),int(getattr(nli.config,'max_position_embeddings',512)));cache={}
        def predict(a,b):
            key=(sha(a),sha(b))
            if key in cache:return cache[key]
            inputs=tokenizer(a,b,truncation=False,return_tensors='pt')
            if inputs['input_ids'].shape[-1]>limit:cache[key]=None;return None
            with torch.inference_mode():values=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in inputs.items()}).logits[0],dim=-1).tolist()
            cache[key]={labels[i]:float(v) for i,v in enumerate(values)};return cache[key]
        records=[];verdict_rows=[]
        for i,out in enumerate(outputs,1):
            phase('diagnostics_'+str(i)+'_of_36');row=by_id[out['id']];verdict=None
            if out['output'].strip() and not out['hit_output_limit']:
                verdict=verifier.verify(row['source'],out['output'],predict)
                verdict_rows.append({**{k:out[k] for k in ('id','model','prompt_arm')},'source_sha256':sha(row['source']),'output_sha256':sha(out['output']),'verifier':verdict})
                write_lines(output/'verifier_results.private.jsonl',verdict_rows)
            records.append(output_metrics(pack,row,out,verdict));write_lines(output/'metrics.private.jsonl',records)
        summary={'protocol':PROTOCOL,'status':'execution_complete_with_generation_failures' if any(not r['available'] for r in records) else 'pilot_complete',
            'train_pairs':12,'development_pairs':4,'training_probes':2,'epochs':4,'updates_per_arm':24,
            'generation_attempts':36,'available_generations':sum(r['available'] for r in records),'adapter_reload_verified':True,
            'training_called':True,'reserved_test_rows_read':0,'semantic_accuracy':None,'author_style_accuracy':None,
            'oracle_plans_are_deployment_result':False,'quality_assessment':'pending; uncalibrated NLI and surface metrics are not a semantic/style grade',
            'development_losses':loss_reports}
        write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_revision':NLI_REVISION,'config':CONFIG,'elapsed_seconds':time.monotonic()-started,'generation_and_training_memory':memory,'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True);phase(summary['status'])
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',completed_generations=len(outputs),exception_type=type(exc).__name__)
        status.update(baseline.safe_failure_details(exc));write_json(output/'run_status.json',status)
        raise RuntimeError('Pilot incomplete; inspect saved status. Exception details suppressed to protect credentials.') from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument('--prepare-pack',action='store_true');a.add_argument('--dry-run',action='store_true');a.add_argument('--execute-training',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.pack is None:p.error('--pack required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_training:run_training(pack,args.output_dir,args.expected_pack_sha);return
    fresh(args.output_dir)
    summary={'protocol':PROTOCOL,'status':'cpu_dry_run_passed' if args.dry_run else 'preflight_passed',
        'train_pairs':12,'development_pairs':4,'updates_per_arm':len(schedules([r['id'] for r in pack['train']])),
        'generation_attempts':len(generation_rows(pack))*len(ARMS)*3,'model_calls':0,'training_called':False,'reserved_test_rows_read':0}
    write_json(args.output_dir/'preflight.json',summary)
    print('CPU preflight passed: 12 training pairs, two fresh adapters, 24 updates each, 36 generations planned. No models, Secrets or GPU accessed.')


if __name__=='__main__':main()
