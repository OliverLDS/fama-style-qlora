"""Private 40-paragraph, two-stage relation-aware annotation pilot builder."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from .author_comparison_corpus import file_sha,words
from .human_prose_annotations import author_key
from fama_style.rhetoric.experiment import fresh,write_json,write_lines

PROTOCOL='relation_pilot_v1'
STRATA=('continuation','scope_transition')
RELATIONS=('premise_consequence','contrast','qualification','restatement','definition_explication',
           'parallel_actions','comparison','causal','conditional','association','temporal','attribution_change','example_result','argument_transition','none','unresolved')
STAGE1_PROMPT='''Create meaning-preserving neutral-English pairs from the attached ORIGINAL paragraphs. These are historical academic economics texts. Work only from the supplied paragraph, not remembered versions, other batches or outside context.

For every item, retain paragraph_id and copy original_paragraph exactly. Rewrite into clear, neutral English for an economically literate reader. Preserve all substantive claims, actors/roles, quantities, quantifiers, negation, uncertainty, attribution, conditions, causal direction, comparisons and inferential force. Preserve hypotheses, conjectures, counterfactuals and findings as such. Do not add missing context, explanations, mechanisms or conclusions. Do not make the neutral version more formal. Natural wording and sentence changes are allowed; there is no edit-distance or sentence-count target. If already clear, keep it unchanged and mark already_clear. If safe rewriting is uncertain, mark pending_review and explain; do not guess.

Record atomic semantic claims C1, C2, ... within each paragraph. Separate distinct propositions even if they occur in one original sentence; do not combine two propositions merely to make alignment easier. Each claim records its proposition, actors/roles, conditions, quantities, negation, modality, attribution and exact evidence quotes from both versions, with one-based occurrence numbers. Record alignment as preserved, omitted, changed or uncertain. These are your annotations, not independent verification.

Segment the complete original into O1, O2, ... and the complete neutral paragraph into S1, S2, ... . exact_text strings, in order, must reconstruct each paragraph apart from whitespace. Sentence claim_ids identify explicitly expressed claim content. reference_only_claim_ids identify back-references without repeating the full proposition; do not count a pronoun as a full repeated claim. Do not force one sentence per claim. If a claim has components distributed across sentences, identify the granularity issue in review_notes rather than silently asserting complete duplication. Flag missing discourse context, ambiguous reference or notation, and uncertain extraction or meaning. Never silently correct original text.

Do NOT label KEEP/MERGE preferences or permissible layouts at this stage. Original sentence organization is reference provenance, not the only valid organization. Later source-only judgments will be collected in a separate payload. No full three-level taxonomy is requested in this bounded pilot.

Use the attached stage1_output.schema.json exactly. Return one UTF-8 JSON object per line, without Markdown fences, preferably as a downloadable JSONL file. Cover every supplied paragraph exactly once, including pending_review items. Do not invent provenance: use null for unknown visible model/effort. paired_context is true and independently_verified is false. Preserve the first complete response; do not optimize for a detector or human-authorship claims. Separately provide a brief coverage/limitations report with IDs covered, counts by rewrite_status and unresolved problems.'''
STAGE2_PROMPT='''SOURCE-ONLY BOUNDARY JUDGMENT (template; run only with the later prepared source inputs).

Use only source_paragraph and source sentence text in the supplied input. Do not retrieve original author paragraphs, previous pairs, reference layouts, paper titles, other chat answers or remembered versions. Do not infer the author. Assess clear, compact academic economics prose; do not imitate a named author or optimize a detector. Shared model family, pretraining and automatic chat retrieval mean this is not certified independent evaluation.

First extract fresh source-only atomic claims D1, D2, ... with exact source evidence and substantive restrictions. Reconstruct each source sentence exactly and distinguish full claim expression from reference_only_claim_ids. Do not use claim IDs or free-text propositions from the paired stage. Claim granularity must preserve distinct quantities, actors, conditions and uncertainty.

For EVERY adjacent source-sentence boundary B1, B2, ... , record left_sentence_id and right_sentence_id; its source-supported relation from the finite schema labels; exact evidence from both sides; and restrictions needed to preserve actors, attribution, modality, negation, conditions, quantities, comparison and causal direction. Do not infer a causal edge just because two sentences are adjacent.

Record permissible as KEEP, MERGE, BOTH or UNRESOLVED. KEEP means only separation is reasonably supported; MERGE means only merging is reasonably supported; BOTH means both are reasonable; UNRESOLVED means the supplied text or available realization evidence is insufficient. Judge reasonable organization AND meaning, not merely whether a semicolon makes a grammatical sentence. Do not force class balance or choose a merge to match any presumed reference. A valid alternative need not reconstruct one author's layout.

Record presentation_preference separately as KEEP, MERGE, NONE or UNRESOLVED, with a rationale and confidence low/medium/high. If MERGE or BOTH is judged permissible, supply one concrete single-sentence merge_candidate covering the two sentences. Retain every claim and qualification. If no safe candidate is available, abstain instead of certifying a merge. For KEEP or UNRESOLVED, merge_candidate may be null. Mark candidate_quality as unverified: it must later be checked against source claims; your own candidate is not independent evidence of fidelity.

Decisions are local to the pair. Several individually acceptable merges do not automatically make a long chained group acceptable. Flag nested scopes, multiple actors/attributions and component-spanning claims. Report external-context needs and uncertainty. Do not change the source.

Use stage2_output.schema.json exactly; return one JSON object per paragraph with complete sentence and boundary coverage, no Markdown fences, and a separate coverage/limitations report. The original author's layout will be retained separately offline. Your verdicts are interpretations, not verified author intent or semantic certificates.'''


def output_schemas():
    string={'type':'string'};nullable={'type':['string','null']}
    arr=lambda x:{'type':'array','items':x}
    obj=lambda p:{'type':'object','properties':p,'required':list(p),'additionalProperties':False}
    enum=lambda values:{'type':'string','enum':list(values)}
    ids=lambda prefix:arr({'type':'string','pattern':'^'+prefix+'[1-9][0-9]*$'})
    ev=arr(obj({'quote':{'type':'string','minLength':1},'occurrence':{'type':'integer','minimum':1}}))
    provenance=obj({'visible_model_label':nullable,'thinking_setting':nullable,'paired_context':{'const':True},'independently_verified':{'const':False}})
    claim_base={'claim_id':{'type':'string','pattern':'^C[1-9][0-9]*$'},'proposition':string,
        'actors_and_roles':arr(string),'conditions':arr(string),'quantities':arr(string),'negation':arr(string),'modality':arr(string),'attribution':arr(string)}
    sentence=lambda prefix,claim_prefix:arr(obj({'sentence_id':{'type':'string','pattern':'^'+prefix+'[1-9][0-9]*$'},
        'exact_text':{'type':'string','minLength':1},'claim_ids':ids(claim_prefix),'reference_only_claim_ids':ids(claim_prefix)}))
    stage1=obj({'annotation_version':{'const':PROTOCOL+'_stage1'},'paragraph_id':string,
        'original_paragraph':{'type':'string','minLength':1},'plain_english':nullable,
        'rewrite_status':enum(('simplified','already_clear','pending_review')),'rewrite_notes':string,
        'claims':arr(obj({**claim_base,'original_evidence':ev,'neutral_evidence':ev,
            'alignment':enum(('preserved','omitted','changed','uncertain'))})),
        'original_sentences':sentence('O','C'),'neutral_sentences':sentence('S','C'),
        'needs_external_context':{'type':'boolean'},'needs_review':{'type':'boolean'},'review_notes':arr(string),'provenance':provenance})
    stage2=obj({'annotation_version':{'const':PROTOCOL+'_stage2'},'paragraph_id':string,
        'source_paragraph':{'type':'string','minLength':1},'source_sentences':sentence('S','D'),
        'source_claims':arr(obj({**claim_base,'claim_id':{'type':'string','pattern':'^D[1-9][0-9]*$'},'source_evidence':ev})),
        'boundaries':arr(obj({'boundary_id':{'type':'string','pattern':'^B[1-9][0-9]*$'},
            'left_sentence_id':string,'right_sentence_id':string,'relation':enum(RELATIONS),'source_evidence':ev,
            'restrictions':arr(string),'permissible':enum(('KEEP','MERGE','BOTH','UNRESOLVED')),
            'presentation_preference':enum(('KEEP','MERGE','NONE','UNRESOLVED')),'rationale':string,
            'confidence':enum(('low','medium','high')),'merge_candidate':nullable,'candidate_quality':{'const':'unverified'}})),
        'needs_external_context':{'type':'boolean'},'needs_review':{'type':'boolean'},'review_notes':arr(string),
        'provenance':obj({'visible_model_label':nullable,'thinking_setting':nullable,'reference_supplied':{'const':False},'independently_verified':{'const':False}})})
    return stage1,stage2


def validate_selection(registry,records,excluded):
    papers=registry['papers'];old=excluded['papers']
    if len(papers)!=4 or len({p['paper_id'] for p in papers})!=4:raise ValueError('four_papers')
    old_titles={re.sub(r'\W','',p['title'].casefold()) for p in old}
    old_authors={author_key(n) for p in old for n in p['authors']};new_authors=set()
    old_hashes={p['pdf_sha256'] for p in old if p.get('pdf_sha256')}
    for p in papers:
        if type(p['publication_year']) is not int or not 1800<=p['publication_year']<2020:raise ValueError('pre2020')
        if not p['publication_year_checked'] or not p['publication_evidence_url'].startswith('https://'):raise ValueError('date_evidence')
        if re.sub(r'\W','',p['title'].casefold()) in old_titles or p['pdf_sha256'] in old_hashes:raise ValueError('existing_paper')
        for n in p['authors']:
            key=author_key(n)
            if key in old_authors or key in new_authors:raise ValueError('author_group_overlap')
            if set(words(n.lower()))&{'fama','french','jensen','blume'}:raise ValueError('project_author')
            new_authors.add(key)
    byid={p['paper_id']:p for p in papers}
    if len(records)!=40 or len({r['paragraph_id'] for r in records})!=40:raise ValueError('forty_unique_records')
    if len({re.sub(r'\s+',' ',r['original_paragraph']).strip() for r in records})!=40:raise ValueError('duplicate_text')
    for r in records:
        if r['paper_id'] not in byid or r['stratum'] not in STRATA:raise ValueError('paper_or_stratum')
        if r['pdf_sha256']!=byid[r['paper_id']]['pdf_sha256']:raise ValueError('pdf_mismatch')
        if hashlib.sha256(r['original_paragraph'].encode()).hexdigest()!=r['original_sha256']:raise ValueError('original_hash')
        if not 40<=len(words(r['original_paragraph']))<=300:raise ValueError('paragraph_budget')
        if r['pdf_page_start']>r['pdf_page_end'] or r['pdf_page_start']<1:raise ValueError('page_location')
    if Counter((r['paper_id'],r['stratum']) for r in records)!={(p,s):5 for p in byid for s in STRATA}:raise ValueError('five_per_stratum_per_paper')


def stage2_payload(stage1_record):
    """Only the actual neutral text: no references, paired claims or preference hints."""
    if not isinstance(stage1_record['plain_english'],str) or not stage1_record['plain_english'].strip():raise ValueError('no_neutral_text')
    return {'paragraph_id':stage1_record['paragraph_id'],'source_paragraph':stage1_record['plain_english']}


def build(registry_path,selection_path,excluded_path,output):
    registry=json.loads(registry_path.read_text());records=[json.loads(l) for l in selection_path.read_text().splitlines()]
    excluded=json.loads(excluded_path.read_text());validate_selection(registry,records,excluded)
    for p in registry['papers']:
        if file_sha(Path(p['pdf']))!=p['pdf_sha256']:raise ValueError('source_pdf_changed')
    fresh(output);(output/'common').mkdir();(output/'responses/stage1').mkdir(parents=True);(output/'responses/stage2').mkdir()
    schema1,schema2=output_schemas()
    write_json(output/'common/stage1_output.schema.json',schema1);write_json(output/'common/stage2_output.schema.json',schema2)
    (output/'common/STAGE1_PROMPT.md').write_text(STAGE1_PROMPT+'\n');(output/'common/STAGE2_PROMPT.md').write_text(STAGE2_PROMPT+'\n')
    bypaper={p['paper_id']:p for p in registry['papers']};key=[]
    for pi,paper in enumerate(sorted(bypaper)):
        selected=[r for s in STRATA for r in records if r['paper_id']==paper and r['stratum']==s]
        for bi in (0,1):
            batch=2*pi+bi+1;rows=selected[bi::2];folder=output/('stage1/batch_%02d'%batch);folder.mkdir(parents=True)
            write_lines(folder/'paragraphs.jsonl',[{'paragraph_id':r['paragraph_id'],'original_paragraph':r['original_paragraph']} for r in rows])
            (folder/'TASK.md').write_text(STAGE1_PROMPT+'\n\nProcess the five attached paragraphs using stage1_output.schema.json. Save responses_%02d.jsonl and coverage_%02d.md.\n'%(batch,batch))
            for r in rows:
                p=bypaper[paper];key.append({'paragraph_id':r['paragraph_id'],'paper_id':paper,'batch':batch,'stratum':r['stratum'],
                    'original_sha256':r['original_sha256'],'pdf_page_start':r['pdf_page_start'],'pdf_page_end':r['pdf_page_end'],
                    'printed_page_start':r['pdf_page_start']+p['printed_page_offset'],'printed_page_end':r['pdf_page_end']+p['printed_page_offset']})
    write_json(output/'PRIVATE_SELECTION.json',key);shutil.copyfile(selection_path,output/'source_selection_and_repairs.private.jsonl')
    write_json(output/'source_registry.private.json',registry)
    summary={'protocol':PROTOCOL,'status':'stage1_ready_stage2_specification_only','papers':4,'paragraphs':40,
        'stage1_batches':8,'paragraphs_per_batch':5,'strata':dict(Counter(r['stratum'] for r in records)),
        'years':[min(p['publication_year'] for p in registry['papers']),max(p['publication_year'] for p in registry['papers'])],
        'word_range':[min(len(words(r['original_paragraph'])) for r in records),max(len(words(r['original_paragraph'])) for r in records)],
        'paragraphs_with_transcription_or_boundary_repairs':sum(r['transcription_audit']['before']!=r['original_paragraph'] for r in records),
        'new_papers_not_new_pretraining_holdout':True,'sampling_strata_not_forced_operation_labels':True,
        'stage2_inputs_created':False,'full_three_level_taxonomy_requested':False,'responses_received':0,
        'new_model_calls':0,'gpu_jobs_started':0,'training_called':False,'reserved_content_used':False}
    write_json(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('''# Relation-aware pilot: start with Stage 1

40 complete paragraphs, ten from each of four new pre-2020 papers. Use GPT-6 with High thinking effort, consistent across batches. Eight Stage 1 batches contain five paragraphs each.

For each stage1/batch_01 through batch_08:
1. Paste TASK.md.
2. Attach ONLY that batch's paragraphs.jsonl and common/stage1_output.schema.json.
3. Save the first complete, unedited responses_01.jsonl through responses_08.jsonl in responses/stage1/. Save any coverage_XX.md there too. Record any retries, changed settings or shared context in SESSION_NOTES.md or your reply.

Do not upload PDFs, private selection/repair records, source metadata, other batch answers, prior classifier outputs or Stage 2 material. Separate chats can reduce direct carryover but cannot guarantee independence from automatic retrieval. Do not share credentials.

Notify Codex when all eight Stage 1 responses are ready. Codex must check coverage, originals, sentence/claim evidence and uncertainty before constructing Stage 2 inputs from actual returned neutral paragraphs. Stage 2 files in common/ are a specification/template only: do not run them yet, and do not substitute the originals for neutral inputs.

Stage 2 will ask for source-only logical relations, permitted KEEP/MERGE/BOTH/UNRESOLVED decisions, separate presentation preferences and concrete merge candidates. Its payload will contain only paragraph_id and source_paragraph, with freshly extracted D-claims; paired references, claims and author layouts stay offline. No sentence-count, edit-distance or label-balance target is imposed.

These are discovery annotations, not approved training data or a reserved benchmark. The pilot does not establish author style or human authorship. No Kaggle run or fine-tuning is needed.
''')
    (output/'SESSION_NOTES.md').write_text('Visible model label:\nThinking effort:\nChat arrangement/shared context:\nRetries, regenerations or edited outputs:\n\nDo not include credentials.\n')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'registry_sha256':file_sha(registry_path),'selection_sha256':file_sha(selection_path),'excluded_source_metadata_sha256':file_sha(excluded_path),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SESSION_NOTES.md'}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','selection','excluded-sources','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.registry,a.selection,a.excluded_sources,a.output),indent=2))


if __name__=='__main__':main()
