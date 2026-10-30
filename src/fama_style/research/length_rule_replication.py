"""Freeze additional-paper sampling and blinded generation for the 30-word rule.

Existing nonreserved neutral paragraphs are reused; no candidate calls, detector
optimization, reserved evaluation or model training are performed here.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .author_comparison_corpus import words
from .author_control_replication import verify_files
from .human_prose_annotations import author_key
from .human_prose_review import normalized, occurrence_span, schema_errors
from .relation_pilot_review import file_manifest, file_sha, read_rows
from .relation_pilot_verification import allocate, output_schema as verification_schema, PROMPT as VERIFICATION_PROMPT
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='fixed_length_replication_v1'
CONFIG={'protocol':PROTOCOL,'cutoff':30,'word_counter':"[A-Za-z]+(?:'[A-Za-z]+)?",
    'source_pool':'earlier_human_prose_training_only','papers':4,'pairs_per_stratum_per_paper':4,
    'strata':['at_most_30','over_30'],'paper_selection':'capacity_then_seeded_hash_then_distinct_authors',
    'pair_selection':'four_word_length_quantiles_within_each_paper_stratum_hash_ties',
    'seed':PROTOCOL,'generation_before_verification':True,'threshold_refitting':False,
    'candidate_or_judge_outcomes_used_for_sampling':False,
    'primary':'faithful_decisive_readability','side_cohorts':['faithful_comparable','claim_shift','unresolved','generation_abstention'],
    'methods':['keep','merge_all_available','fixed_30_available'],
    'availability_policy':'an_abstained_generation_is_unavailable_and_forces_keep; report_coverage_separately',
    'reporting':'counts_and_macro_paper_scores_separate_side_cohorts; sampling_balanced_not_population_estimate'}
GENERATION_PROMPT='''Create one conservative single-sentence replacement for each supplied adjacent sentence pair in its source paragraph.

Use only source_paragraph, left_sentence and right_sentence. The pair is fixed;
do not choose a different pair. Preserve every substantive claim, actor and role,
quantity/quantifier, condition, exception, negation, modality, attribution, question
or hypothesis status, and causal/inferential direction. Do not add a relation
merely to join clauses, and do not strengthen a tentative or conditional claim.
Use surrounding source only to resolve an unambiguous referent needed by the pair;
record exact source_context_evidence when used. Do not import other paragraph
claims, outside knowledge, remembered articles or earlier answers. A repeated
actor name may be replaced only if its identity and scope remain unambiguous.

Produce one natural, readable sentence if feasible. Semicolons are allowed but
do not by themselves make a useful merge. Do not force brevity or target a word
count, edit distance, author style or preferred sentence layout. If no adequately
faithful single sentence can be produced, set generation_status to abstained,
candidate to null and explain why. Otherwise use generated and an empty
abstention_reason. Do not silently drop difficult cases or return several options.

Do NOT judge whether merging is preferable to separation, score fidelity, produce
a taxonomy, infer the author, or supply a revised paragraph. A later assessment
will examine the fixed candidate; generation is not semantic certification.
Copy item_id, source_paragraph, left_sentence and right_sentence exactly. Follow
generation_output.schema.json and return one downloadable UTF-8 JSON object per
line without Markdown fences. Cover every supplied item exactly once, including
abstentions, and provide a separate coverage/limitations report. Preserve the
first complete output. Use null for unknown visible model/effective effort;
requested effort is not necessarily effective effort. earlier_judgments_supplied
and independently_verified are false. Same-model pretraining and possible chat
retrieval prevent any guarantee of independent context.
'''


def generation_schema():
    string={'type':'string'}
    obj=lambda p:{'type':'object','properties':p,'required':list(p),'additionalProperties':False}
    evidence={'type':'array','items':obj({'quote':{'type':'string','minLength':1},'occurrence':{'type':'integer','minimum':1}})}
    return obj({'annotation_version':{'const':PROTOCOL+'_generation'},'item_id':string,
        **{k:{'type':'string','minLength':1} for k in ('source_paragraph','left_sentence','right_sentence')},
        'candidate':{'type':['string','null']},'generation_status':{'type':'string','enum':['generated','abstained']},
        'abstention_reason':string,'source_context_evidence':evidence,
        'provenance':obj({'visible_model_label':{'type':['string','null']},'thinking_setting':{'type':['string','null']},
            'earlier_judgments_supplied':{'const':False},'independently_verified':{'const':False}})})


def check_generation(row,expected):
    errors=schema_errors(row,generation_schema())
    if errors:return errors
    for key in ('item_id','source_paragraph','left_sentence','right_sentence'):
        if row[key]!=expected[key]:errors.append({'code':'fixed_input_changed','field':key})
    if row['generation_status']=='generated':
        if not isinstance(row['candidate'],str) or not row['candidate'].strip() or row['abstention_reason'].strip():
            errors.append({'code':'inconsistent_generated_candidate'})
    elif row['candidate'] is not None or not row['abstention_reason'].strip():
        errors.append({'code':'inconsistent_abstention'})
    for evidence in row['source_context_evidence']:
        if occurrence_span(row['source_paragraph'],evidence) is None:errors.append({'code':'invalid_context_evidence'})
    return errors


def verification_payload(row):
    if row['generation_status']!='generated' or not isinstance(row['candidate'],str) or not row['candidate'].strip():
        raise ValueError('no_actual_generated_candidate')
    return {k:row[k] for k in ('item_id','source_paragraph','left_sentence','right_sentence','candidate')}


def digest(value):return hashlib.sha256((PROTOCOL+'|'+value).encode()).hexdigest()


def source_pool(audit,registry):
    """Filter training metadata before deserializing annotations; no target use."""
    allowed={p['paper_id'] for p in registry['papers'] if p['split']=='training'}
    pool=[];excluded=Counter()
    with (audit/'audits.private.jsonl').open() as stream:
        training_lines=[line for line in stream if re.search(r'"split"\s*:\s*"training"',line)]
    for line in training_lines:
        row=json.loads(line)
        if row['split']!='training':continue
        if row['paper_id'] not in allowed:raise ValueError('training_paper_registry_mismatch')
        a=row['annotation'];mechanics=row['audit']
        if not row['schema_valid'] or mechanics['errors'] or mechanics['program_errors']:
            excluded['mechanically_invalid']+=1;continue
        if a['rewrite_status']=='pending_review' or not a.get('plain_english') or not a['source_schema'] or a['source_quality']['status']=='unusable':
            excluded['unavailable_neutral_source']+=1;continue
        sentences=[s['exact_text'] for s in a['source_schema']['sentences']]
        if normalized(' '.join(sentences))!=normalized(a['plain_english']):raise ValueError('source_sentence_coverage')
        for i,(left,right) in enumerate(zip(sentences,sentences[1:]),1):
            count=len(words(left))+len(words(right));key=row['paragraph_id']+':B'+str(i)
            pool.append({'paper_id':row['paper_id'],'paragraph_id':row['paragraph_id'],'boundary_id':'B'+str(i),
                'source_paragraph':a['plain_english'],'left_sentence':left,'right_sentence':right,
                'item_id':'FL_'+digest(key)[:16],'pair_words':count,'rule_merge':count<=30,
                'stratum':'at_most_30' if count<=30 else 'over_30',
                'source_quality_status':a['source_quality']['status'],
                'upstream_source_issues':a['source_quality']['issues'],
                'upstream_needs_review':a['needs_review'],
                'upstream_needs_external_context':a['source_quality']['needs_external_context']})
    return pool,dict(excluded)


def select(pool,registry,prior_papers):
    byid={p['paper_id']:p for p in registry['papers']}
    forbidden={p['paper_id'] for p in prior_papers['papers']}
    authors={author_key(a) for p in prior_papers['papers'] for a in p['authors']}
    capacities={p:{s:sum(r['paper_id']==p and r['stratum']==s for r in pool) for s in CONFIG['strata']} for p in sorted({r['paper_id'] for r in pool})}
    eligible=[p for p,c in capacities.items() if min(c.values())>=4 and p not in forbidden]
    chosen=[]
    for pid in sorted(eligible,key=digest):
        p=byid[pid];names={author_key(a) for a in p['authors']}
        if p['split']!='training':raise ValueError('nontraining_paper')
        year=p['publication_year']
        if type(year) is not int or not 1800<=year<2020 or not p['publication_evidence_url'].startswith('https://'):
            raise ValueError('pre2020_metadata')
        if names & authors:continue
        chosen.append(pid);authors.update(names)
        if len(chosen)==4:break
    if len(chosen)!=4:raise ValueError('insufficient_additional_papers_with_both_strata')
    selected=[]
    for paper in chosen:
        for stratum in CONFIG['strata']:
            ordered=sorted([r for r in pool if r['paper_id']==paper and r['stratum']==stratum],key=lambda r:(r['pair_words'],digest(r['item_id'])))
            for i in range(4):
                quantile=ordered[i*len(ordered)//4:(i+1)*len(ordered)//4]
                selected.append(quantile[len(quantile)//2])
    if len({r['item_id'] for r in selected})!=32:raise ValueError('duplicate_selected_pair')
    return selected,capacities


def build(audit,registry_path,selector,prior_papers_path,output):
    verify_files(audit);verify_files(selector)
    earlier=json.loads((selector/'summary.json').read_text())
    if {v['length_only'] for v in earlier['selected_thresholds'].values()}!={30}:raise ValueError('original_cutoff_changed')
    registry=json.loads(registry_path.read_text());prior=json.loads(prior_papers_path.read_text())
    if set(earlier['per_paper'])!={p['paper_id'] for p in prior['papers']}:raise ValueError('prior_paper_ids_changed')
    pool,excluded=source_pool(audit,registry);selected,capacities=select(pool,registry,prior)
    jobs=[{'paragraph_id':r['paragraph_id'],'paper_id':r['paper_id'],
           'payload':{k:r[k] for k in ('item_id','source_paragraph','left_sentence','right_sentence')}} for r in selected]
    batches=allocate(jobs);fresh(output);(output/'common').mkdir();(output/'responses/generation').mkdir(parents=True);(output/'responses/verification').mkdir()
    (output/'common/GENERATION_PROMPT.md').write_text(GENERATION_PROMPT)
    write_json(output/'common/generation_output.schema.json',generation_schema())
    # Reuse the fixed-candidate verification specification, without placeholder inputs.
    (output/'common/VERIFICATION_PROMPT_SPECIFICATION.md').write_text(VERIFICATION_PROMPT)
    write_json(output/'common/verification_output.schema.json',verification_schema())
    assignment={}
    for bi,batch in enumerate(batches,1):
        folder=output/('generation/batch_%02d'%bi);folder.mkdir(parents=True)
        write_lines(folder/'items.jsonl',[j['payload'] for j in batch])
        (folder/'TASK.md').write_text(GENERATION_PROMPT+'\nProcess all %d supplied pairs; save responses_%02d.jsonl and coverage_%02d.md.\n'%(len(batch),bi,bi))
        assignment.update({j['payload']['item_id']:bi for j in batch})
    write_lines(output/'PRIVATE_SELECTION.jsonl',[{**r,'batch':assignment[r['item_id']]} for r in selected])
    write_json(output/'PRIVATE_SAMPLING.json',{'candidate_pair_capacities':capacities,'excluded_source_paragraph_counts':excluded})
    chosen_ids={r['paper_id'] for r in selected};papers=[p for p in registry['papers'] if p['paper_id'] in chosen_ids]
    write_json(output/'sources.json',{'papers':papers})
    write_json(output/'protocol.json',CONFIG)
    summary={'protocol':PROTOCOL,'status':'generation_ready_verification_specification_only','items':len(selected),
        'papers':[p['paper_id'] for p in papers],'paragraphs':len({r['paragraph_id'] for r in selected}),
        'strata':dict(Counter(r['stratum'] for r in selected)),'generation_batches':len(batches),
        'batch_sizes':[len(b) for b in batches],'pair_word_range':[min(r['pair_words'] for r in selected),max(r['pair_words'] for r in selected)],
        'source_quality_status':dict(Counter(r['source_quality_status'] for r in selected)),
        'upstream_needs_review_pairs':sum(r['upstream_needs_review'] for r in selected),
        'upstream_context_dependent_pairs':sum(r['upstream_needs_external_context'] for r in selected),
        'prior_project_exposure':True,'papers_disjoint_from_cutoff_development':True,
        'fresh_candidates_not_fresh_paragraphs':True,'sampling_balanced_not_natural_prevalence':True,
        'cutoff_refitted':False,'cutoff_decisions_exported':False,'earlier_claims_or_judgments_exported':False,
        'reserved_content_used':False,'generation_responses_received':0,'verification_inputs_created':False,
        'new_model_calls':0,'gpu_jobs_started':0,'training_called':False}
    write_json(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('''# Fixed-rule replication: candidate generation first

32 pairs from four additional, nonreserved papers, in eight batches of four.
Use GPT-6 and request High thinking effort; report uncertainty if effective effort
is unclear. For generation/batch_01 through batch_08, paste TASK.md and attach
ONLY its items.jsonl plus common/generation_output.schema.json.

Save the first complete unedited responses_01.jsonl through responses_08.jsonl
and coverage_XX.md in responses/generation/. Notify Codex when all eight are ready.
Record known retries/settings/context changes in SESSION_NOTES.md or your reply.
Unknown details may remain unknown. Do not share credentials.

Do not upload this file, PRIVATE_SELECTION, PRIVATE_SAMPLING, protocol, sources,
earlier answers, labels, reference paragraphs or the verification specification.
Only actual neutral source text and the fixed local pair reach the generator.
The generator creates one candidate or abstains; it does not choose pairs or
judge whether merging is preferable. Do not retry to obtain a desired outcome.

These neutral paragraphs were already used in earlier training-pool research.
Their papers differ from those used to develop this rule, but they are not
project-unseen papers or paragraphs. Source/context cautions remain offline.
No reserved or development-split paragraphs are used. Local checks are relative
to supplied neutral text, not proof of fidelity to the original author's text.

After generation is audited, Codex will prepare fixed-candidate verification
inputs from actual returned candidates for all available items, not only a
preferred subset. Files in common/ describe that later specification only.
Do not run verification yet, substitute originals or fabricate missing candidates.
Separate chats cannot guarantee independence from pretraining or automatic
retrieval. No classifier fit, threshold tuning or Kaggle job is needed.
''')
    (output/'SESSION_NOTES.md').write_text('Visible model:\nRequested effort:\nKnown effective effort or uncertain:\nChat/context arrangement if known:\nRetries/edits if known:\n\nNo credentials.\n')
    file_manifest(output,protocol=PROTOCOL,code_sha256=file_sha(Path(__file__)),
        upstream_audit_manifest_sha256=file_sha(audit/'manifest.json'),registry_sha256=file_sha(registry_path),
        selector_manifest_sha256=file_sha(selector/'manifest.json'),prior_papers_sha256=file_sha(prior_papers_path))
    manifest=json.loads((output/'manifest.json').read_text());manifest['files'].pop('SESSION_NOTES.md');write_json(output/'manifest.json',manifest)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','registry','selector','prior-papers','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.audit,a.registry,a.selector,a.prior_papers,a.output),indent=2))


if __name__=='__main__':main()
