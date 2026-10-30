"""Prepare a private paired source-generation pilot; no model or detector calls.

Stage 1 extracts semantic briefs. Reference-free Stage 2 handoffs are built only
after responses arrive and their alignment is assessed. All prose stays private.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .author_control_replication import verify_files

PROTOCOL = 'neutral_reconstruction_v1'
BRIEF_VERSION = 'economic_content_brief_v1'
GROUPS = ('Shiller', 'Holmstrom', 'Baker_Wurgler')
CLAIM_LISTS = ('actor_ids','quantities','quantifiers','comparisons','negation','modality',
               'conditions','exceptions','attribution','temporal_scope','causal_direction')
CLAIM_KEYS = {'claim_id','proposition','predicate'} | set(CLAIM_LISTS)
BRIEF_KEYS = {'brief_version','record_id','actors','claims','supported_relationships',
              'essential_terms','citations','context_gaps','needs_review'}
RELATIONS = {'causes','enables','prevents','associated_with','contrasts_with','conditions',
             'supports','explains','defines','compares'}
CONTEXT = re.compile(r'^(?:This|These|Such|It|They|In this section|The former|The latter)\b|\b(?:this chain|these declines|the foregoing|as discussed above|the preceding section)\b',re.I)
CONFIG = {'protocol':PROTOCOL,'status':'prepared_not_executed','examples':12,'papers':6,
          'examples_per_paper':2,'papers_per_group':2,'examples_per_group':4,
          'word_range':[80,200],'selection':'fixed SHA256 rank after source-only quality filters',
          'source_filters':['no recorded quality flags','single PDF page and reading region',
                            'no duplicate_of','no explicit dangling-context pattern'],
          'methods':['direct_rewrite','neutral_reconstruction'],
          'stage1_batches':2,'examples_per_batch':6,'eventual_generated_paragraphs':24,
          'new_paper_role':'training_side_method_development_only; ineligible for fresh held-out evaluation',
          'old_splits_modified':False,'brief_gate':'mechanical validation plus automated source/brief alignment assessment before handoff',
          'handoff':'strict field allowlist; deterministic opaque claim IDs and shuffled claim order; no original or author metadata',
          'long_reference_span_gate_tokens':12,'claimed_chat_independence':False,
          'requested_model':'GPT-6','requested_thinking_effort':'High; record actual visible setting',
          'detector_selection_or_reward':False,'human_review_required':False,
          'model_calls':0,'gpu_jobs':0,'training_pairs_created':0}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rank(text):
    return sha(PROTOCOL+'|'+text)


def eligible(row):
    return (row['status']=='annotation_candidate' and not row['quality_flags']
            and 80<=row['word_count']<=200 and not row['duplicate_of']
            and row['pdf_page_start']==row['pdf_page_end'] and row['column_start']==row['column_end']
            and row['existing_split'] in ('training','train','unassigned_new_acquisition')
            and not CONTEXT.search(row['original_paragraph']))


def select(rows):
    by_group=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['style_group'] in GROUPS and eligible(row):
            by_group[row['style_group']][row['paper_id']].append(row)
    batches=[[],[]]
    commitments=[]
    for group in GROUPS:
        papers=sorted([p for p,rs in by_group[group].items() if len(rs)>=2],key=rank)[:2]
        if len(papers)!=2:raise ValueError('insufficient_eligible_papers:'+group)
        for paper in papers:
            selected=sorted(by_group[group][paper],key=lambda r:rank(r['paragraph_id']))[:2]
            commitments.append({'paper_id':paper,'style_group':group,
                                'previous_role':selected[0]['existing_split'],
                                'committed_role':'training_side_method_development_only',
                                'fresh_heldout_eligible':False})
            for i,row in enumerate(selected):batches[i].append(row)
    # Each batch has one paragraph per paper and two per author group.
    return [sorted(batch,key=lambda r:rank('batch|'+r['paragraph_id'])) for batch in batches],commitments


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))


def require_keys(value,keys,label):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('unexpected_fields:'+label)


def text_list(value,label):
    if not isinstance(value,list) or any(not isinstance(v,str) or not v.strip() for v in value):
        raise ValueError('expected_string_list:'+label)


def validate_brief(brief):
    require_keys(brief,BRIEF_KEYS,'brief')
    if brief['brief_version']!=BRIEF_VERSION or not isinstance(brief['record_id'],str):raise ValueError('brief_identity')
    if not isinstance(brief['needs_review'],bool):raise ValueError('needs_review_boolean')
    for key in ('essential_terms','citations','context_gaps'):text_list(brief[key],key)
    if any(len(term.split())>8 for term in brief['essential_terms']):raise ValueError('essential_term_is_not_a_short_anchor')
    if not isinstance(brief['actors'],list) or not isinstance(brief['claims'],list) or not brief['claims']:
        raise ValueError('actors_and_nonempty_claims_required')
    actor_ids=set()
    for actor in brief['actors']:
        require_keys(actor,{'actor_id','name'},'actor')
        if not all(isinstance(v,str) and v.strip() for v in actor.values()) or actor['actor_id'] in actor_ids:raise ValueError('actor_identity')
        actor_ids.add(actor['actor_id'])
    claim_ids=set()
    for claim in brief['claims']:
        require_keys(claim,CLAIM_KEYS,'claim')
        if not all(isinstance(claim[key],str) and claim[key].strip() for key in ('claim_id','predicate','proposition')):raise ValueError('claim_text')
        if claim['claim_id'] in claim_ids:raise ValueError('duplicate_claim_id')
        claim_ids.add(claim['claim_id'])
        for key in CLAIM_LISTS:text_list(claim[key],key)
        if not set(claim['actor_ids'])<=actor_ids:raise ValueError('unknown_actor_id')
    if not isinstance(brief['supported_relationships'],list):raise ValueError('relationship_list_required')
    for relation in brief['supported_relationships']:
        require_keys(relation,{'from_claim','to_claim','type','condition','qualification'},'relationship')
        if relation['from_claim'] not in claim_ids or relation['to_claim'] not in claim_ids:raise ValueError('unknown_claim_id')
        if relation['type'] not in RELATIONS:raise ValueError('unsupported_relationship_label')
        if not isinstance(relation['condition'],str) or not isinstance(relation['qualification'],str):raise ValueError('relationship_qualifiers')


def make_handoff(brief,original):
    """Mechanical transport check, not a semantic-approval gate or execution tool."""
    validate_brief(brief)
    if brief['needs_review'] or brief['context_gaps']:raise ValueError('unresolved_content_brief')
    # Construct anew: never copy an envelope containing source/audit metadata.
    handoff=json.loads(json.dumps({key:brief[key] for key in BRIEF_KEYS}))
    mapping={c['claim_id']:'K_'+rank(brief['record_id']+'|'+c['claim_id'])[:12] for c in handoff['claims']}
    for claim in handoff['claims']:claim['claim_id']=mapping[claim['claim_id']]
    handoff['claims'].sort(key=lambda c:c['claim_id'])
    for relation in handoff['supported_relationships']:
        relation['from_claim']=mapping[relation['from_claim']];relation['to_claim']=mapping[relation['to_claim']]
    handoff['supported_relationships'].sort(key=lambda r:rank(json.dumps(r,sort_keys=True)))
    handoff['actors'].sort(key=lambda a:rank(a['actor_id']))
    # Meaning-bearing technical anchors may remain; long reference wording is held.
    tokens=lambda s:re.findall(r"[A-Za-z0-9]+",s.casefold())
    source=tokens(original)
    source_spans={tuple(source[i:i+12]) for i in range(max(0,len(source)-11))}
    def leaves(value):
        if isinstance(value,str):yield value
        elif isinstance(value,dict):
            for v in value.values():yield from leaves(v)
        elif isinstance(value,list):
            for v in value:yield from leaves(v)
    for value in leaves(handoff):
        seq=tokens(value)
        if any(tuple(seq[i:i+12]) in source_spans for i in range(max(0,len(seq)-11))):
            raise ValueError('long_reference_wording_in_brief')
    return handoff


def validate_response(row,source):
    require_keys(row,{'record_id','original_sha256','source_audit','brief','self_check'},'response')
    if row['record_id']!=source['record_id'] or row['original_sha256']!=source['original_sha256']:
        raise ValueError('response_source_identity_mismatch')
    audit=row['source_audit']
    require_keys(audit,{'status','transcription_verification','issues'},'source_audit')
    if audit['status'] not in ('usable','uncertain','unusable') or audit['transcription_verification']!='not_performed':
        raise ValueError('source_audit_status')
    if not isinstance(audit['issues'],list):raise ValueError('source_issue_list')
    for issue in audit['issues']:
        require_keys(issue,{'type','evidence','note'},'source_issue')
        if issue['type'] not in ('transcription','boundary','missing_context','technical_meaning','other'):
            raise ValueError('unknown_source_issue_type')
        if not all(isinstance(issue[key],str) for key in ('evidence','note')):raise ValueError('source_issue_text')
    checks={'all_claims_included','no_added_claims_or_relations','logical_force_preserved','uncertainty_reported'}
    require_keys(row['self_check'],checks,'self_check')
    if not all(isinstance(v,bool) for v in row['self_check'].values()):raise ValueError('self_check_boolean')
    if audit['status']=='unusable':
        if row['brief'] is not None:raise ValueError('unusable_source_must_not_have_brief')
    else:
        validate_brief(row['brief'])
        if row['brief']['record_id']!=source['record_id']:raise ValueError('brief_record_mismatch')
    return {'mechanically_valid':True,'semantic_alignment_assessed':False,
            'eligible_for_semantic_assessment':audit['status']=='usable' and all(row['self_check'].values())
                and row['brief'] is not None and not row['brief']['needs_review'] and not row['brief']['context_gaps']}


def build(root,output):
    root=root.resolve()
    if output.exists():raise ValueError('refuse_to_overwrite_pilot_package')
    corpus=root/'data/author_style_corpus_v1'
    verify_files(corpus)
    rows=[json.loads(line) for line in (corpus/'paragraphs.private.jsonl').read_text().splitlines()]
    batches,commitments=select(rows)
    output.mkdir(parents=True)
    (output/'stage1').mkdir();(output/'private_key').mkdir();(output/'prompts').mkdir()
    keys=[];index=[]
    for number,batch in enumerate(batches,1):
        payload=[]
        for row in batch:
            record_id='NR_'+rank(row['paragraph_id'])[:16]
            payload.append({'record_id':record_id,'original_paragraph':row['original_paragraph'],
                            'original_sha256':row['original_sha256']})
            keys.append({'record_id':record_id,'batch':number,'source_paragraph_id':row['paragraph_id'],
                         'paper_id':row['paper_id'],'style_group':row['style_group'],'authors':row['authors'],
                         'original_sha256':row['original_sha256'],'paper_pdf_sha256':row['paper_pdf_sha256'],
                         'page_location':row['page_location'],'line_refs':row['line_refs'],'word_count':row['word_count'],
                         'original_paragraph':row['original_paragraph'],'source_status':'unverified_mechanically_screened_candidate'})
        path=output/'stage1'/f'inputs_{number:02d}.jsonl';write_rows(path,payload)
        index.append({'batch':number,'input_file':str(path.relative_to(output)),'expected_records':6,
                      'record_ids':[r['record_id'] for r in payload],
                      'save_response_as':f'responses/stage1_briefs/briefs_{number:02d}.jsonl'})
    write_rows(output/'private_key'/'selection.private.jsonl',keys)
    write_json(output/'paper_role_commitments.private.json',commitments)
    write_json(output/'batch_index.json',index)
    write_json(output/'protocol.json',CONFIG)
    for name in ('STAGE1_PROMPT.md','DIRECT_REWRITE_PROMPT.md','RECONSTRUCTION_PROMPT.md','stage1_output.schema.json'):
        (output/'prompts'/name).write_bytes((root/'docs/neutral_reconstruction_v1'/name).read_bytes())
    for folder in ('stage1_briefs','stage2_direct','stage2_reconstruction','stage3_verification'):
        target=output/'responses'/folder;target.mkdir(parents=True)
        (target/'README.txt').write_text('Save first raw JSONL responses here. No credentials. Do not edit earlier responses; use a new version for corrections.\n')
    write_json(output/'session_notes.json',{'requested_model':'GPT-6','requested_thinking_effort':'High',
        'actual_user_reported_model':None,'actual_visible_effort':None,'conversation_arrangement':None,
        'cross_chat_memory_or_retrieval':'uncontrolled','independence_established':False,
        'instructions':'Record actual settings when responses arrive; unknown is valid. No credentials.'})
    summary={'protocol':PROTOCOL,'status':'stage1_prepared_not_executed','examples':len(keys),
             'papers':len(commitments),'examples_per_group':dict(Counter(k['style_group'] for k in keys)),
             'paper_ids':[c['paper_id'] for c in commitments],'batches':2,'examples_per_batch':6,
             'selection_original_words':sum(k['word_count'] for k in keys),
             'selected_papers_committed_training_side':True,'existing_splits_changed':False,
             'all_selected_source_flags_empty':True,'transcription_independently_verified':False,
             'stage2_payloads_built':False,'generations_observed':0,'eventual_generation_count':24,
             'detector_calls':0,'model_calls':0,'GPU_jobs':0,'reserved_content_used':False,'training_pairs_created':0}
    write_json(output/'summary.json',summary)
    (output/'START_HERE.md').write_bytes((root/'docs/neutral_reconstruction_v1/START_HERE.md').read_bytes())
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*')
           if p.is_file() and 'responses' not in p.relative_to(output).parts and p.name!='session_notes.json'}
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'files':files,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'curated_manifest_sha256':hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest(),
        'mutable_unhashed':['session_notes.json','responses/']})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.root,args.output),indent=2))


if __name__=='__main__':main()
