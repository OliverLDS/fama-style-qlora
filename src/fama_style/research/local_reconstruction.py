"""Local loopback-only 1B reconstruction diagnostic; no training or ML imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha

PROTOCOL='local_reconstruction_1b_v1'
MODEL='ggml-org/gemma-3-1b-it-GGUF:Q4_K_M'
METHODS=('direct_rewrite','detailed_brief','compact_claims')
PARAMS={'temperature':0.0,'seed':1729,'max_tokens':512,'top_k':0,'top_p':1.0,
        'min_p':0.0,'repeat_penalty':1.0,'stream':False,'cache_prompt':False}
INSTRUCTION='''Write one clear, neutral economics paragraph for an economically literate
reader who is not a specialist in this paper's methods. Use natural organization.
Preserve every substantive claim, actor and role, quantity, comparison, condition,
exception, negation, uncertainty, attribution, time scope and causal direction.
Retain supporting claims, analogies, technical distinctions, names and citations.
Do not add facts, mechanisms, examples, conclusions or evidentiary disclaimers.
Use qualifiers to scope the claims rather than turning annotation guardrails into
extra commentary. Do not present an expectation as an established finding.
Return only the paragraph: no JSON, headings, Markdown or explanation. If missing
context prevents a faithful paragraph, return BLOCKED: followed by the reason.
'''


def compact(brief):
    """Flatten validated content without dropping nonempty semantic qualifiers."""
    actors={a['actor_id']:a['name'] for a in brief['actors']}
    lines=['Claim list (IDs and order prescribe no writing order):']
    ignored={'claim_id','actor_ids','predicate','proposition'}
    for claim in brief['claims']:
        lines.append(claim['claim_id']+' | Actors: '+', '.join(actors[k] for k in claim['actor_ids']))
        lines.append('Claim: '+claim['proposition'])
        for key,value in claim.items():
            if key not in ignored and value:lines.append(key+': '+json.dumps(value,ensure_ascii=False))
    if brief['supported_relationships']:
        lines.append('Supported relationships (do not infer additional links):')
        for edge in brief['supported_relationships']:lines.append(json.dumps(edge,sort_keys=True,ensure_ascii=False))
    for key in ('essential_terms','citations','context_gaps'):
        if brief[key]:lines.append(key+': '+json.dumps(brief[key],ensure_ascii=False))
    return '\n'.join(lines)


def make_jobs(base_jobs):
    by={}
    for job in base_jobs:by.setdefault(job['record_id'],{})[job['method']]=job['source']
    if len(base_jobs)!=6 or len(by)!=3 or any(set(g)!={'direct_rewrite','neutral_reconstruction'} for g in by.values()):
        raise ValueError('expected_three_frozen_pairs')
    jobs=[]
    for i,(rid,inputs) in enumerate(sorted(by.items())):
        payload=json.loads(inputs['neutral_reconstruction'])
        sources={'direct_rewrite':inputs['direct_rewrite'],'detailed_brief':inputs['neutral_reconstruction'],
                 'compact_claims':compact(payload['content_brief'])}
        for method in METHODS[i:]+METHODS[:i]:
            task=('Rewrite only the supplied original paragraph. Do not force wording changes where unsafe.' if method=='direct_rewrite'
                  else 'Reconstruct only from the supplied content. Do not infer causal relationships from array or list order.')
            prompt=INSTRUCTION+'\n'+task+'\n\nInput:\n'+sources[method]
            jobs.append({'job_id':rid+'_'+method,'record_id':rid,'method':method,'prompt':prompt,
                         'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    return jobs


def prepare(parent,output):
    if output.exists():raise ValueError('refuse_overwrite_packet')
    verify_files(parent)
    jobs=make_jobs(rows(parent/'upload/reconstruction_4b_jobs.jsonl'))
    output.mkdir(parents=True)
    write_rows(output/'jobs.private.jsonl',jobs)
    (output/'references.private.jsonl').write_bytes((parent/'private_key/references.private.jsonl').read_bytes())
    write_json(output/'protocol.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','jobs':9,
        'cases':3,'conditions':METHODS,'model':MODEL,'parameters':PARAMS,
        'reserved_content_used':False,'training_called':False,
        'compact_policy':'Drop empty fields and predicate labels; resolve actor IDs to names; retain every proposition, nonempty semantic qualifier, edge, term and citation.',
        'comparison_limit':'Prose-only contract in all nine jobs; new model, quantization, engine and contract prevent direct causal comparison with Kaggle 4B.',
        'purpose':'Prompt debugging on the local 1B; no eligibility gate or performance claim for 4B.',
        'parent_manifest_sha256':file_sha(parent/'manifest.json')})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})


def call(base,path,payload=None,timeout=600):
    parsed=urlparse(base)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password:
        raise ValueError('loopback_only_endpoint_required')
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
    request=urllib.request.Request(base.rstrip('/')+'/'+path,data=data,headers={'Content-Type':'application/json'})
    # Explicitly bypass proxy settings for local private prose.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=timeout) as response:return json.load(response)


def snapshot(base):
    health=call(base,'health');models=call(base,'v1/models');props=call(base,'props')
    if health.get('status')!='ok' or [r['id'] for r in models['data']]!=[MODEL]:raise ValueError('unexpected_local_model_or_health')
    return {'health':health,'models':models,'props':props}


def preflight(base,jobs,context):
    audits=[]
    for job in jobs:
        messages=[{'role':'user','content':job['prompt']}]
        rendered=call(base,'apply-template',{'messages':messages,'add_generation_prompt':True})['prompt']
        tokens=call(base,'tokenize',{'content':rendered,'add_special':False,'parse_special':True})['tokens']
        if len(tokens)+PARAMS['max_tokens']+8>context:raise ValueError('context_budget_exceeded_no_generation_started')
        audits.append({'job_id':job['job_id'],'input_tokens_preflight':len(tokens),
                       'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
    return audits


def run(packet,output,base='http://127.0.0.1:8080'):
    if output.exists():raise ValueError('refuse_overwrite_run')
    verify_files(packet);jobs=rows(packet/'jobs.private.jsonl')
    if len(jobs)!=9 or len({r['job_id'] for r in jobs})!=9:raise ValueError('nine_jobs_required')
    output.mkdir(parents=True)
    state={'protocol':PROTOCOL,'complete':False,'completed_jobs':0,'training_called':False,'phase':'preflight'}
    write_json(output/'state.json',state);start=time.monotonic()
    try:
        info=snapshot(base);props=info['props'];context=props['default_generation_settings']['n_ctx']
        write_json(output/'server_snapshot.private.json',info)
        model_path=Path(props.get('model_path',''))
        model_hash=None
        if model_path.is_file():
            digest=hashlib.sha256()
            with model_path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            model_hash=digest.hexdigest()
        audits=preflight(base,jobs,context);write_json(output/'token_audit.json',audits)
        write_json(output/'runtime_metadata.json',{'model':MODEL,'model_file_sha256':model_hash,
            'model_file_hash_note':'Unavailable if server path is not readable.' if model_hash is None else 'Hashed loaded model path from props.',
            'chat_template_sha256':hashlib.sha256(props['chat_template'].encode()).hexdigest(),
            'build_info':props.get('build_info'),'context_per_slot':context,'slots':props.get('total_slots'),
            'parameters':PARAMS,'packet_manifest_sha256':file_sha(packet/'manifest.json'),
            'runner_sha256':file_sha(Path(__file__)),'requests_serial':True,'messages_per_request':1,
            'training_called':False,'hardware_backend':'Existing local server; CPU/Metal placement not established by these endpoints.'})
        print('Preflight passed for nine prompts; starting local 1B requests.',flush=True)
        with (output/'generations.private.jsonl').open('x') as stream:
            for job in jobs:
                state['phase']='generating';write_json(output/'state.json',state);before=time.monotonic()
                response=call(base,'v1/chat/completions',PARAMS|{'model':MODEL,'messages':[{'role':'user','content':job['prompt']}]})
                choice=response['choices'][0];text=choice['message']['content']
                if not isinstance(text,str):raise ValueError('missing_text')
                row={k:job[k] for k in ('job_id','record_id','method','prompt_sha256')}
                row.update(raw_response=response,text=text,finish_reason=choice.get('finish_reason'),
                           elapsed_seconds=time.monotonic()-before,
                           status='blocked' if text.strip().startswith('BLOCKED:') else 'generated' if text.strip() else 'empty')
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                state['completed_jobs']+=1;write_json(output/'state.json',state)
                print('Saved %d/9 local outputs'%state['completed_jobs'],flush=True)
        end=snapshot(base)
        identity=lambda value:[(r['id'],r.get('meta')) for r in value['models']['data']]
        if identity(end)!=identity(info) or end['props']['chat_template']!=props['chat_template']:
            raise ValueError('server_identity_changed_during_run')
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-start)
        write_json(output/'state.json',state)
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    except Exception as error:
        state.update(complete=False,failed_phase=state['phase'],phase='failed',error_type=type(error).__name__)
        write_json(output/'state.json',state)
        raise RuntimeError('Local diagnostic failed; inspect state.json. Error type: '+type(error).__name__) from None


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8080')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.parent,args.output)
    else:run(args.packet,args.output,args.endpoint)


if __name__=='__main__':main()
