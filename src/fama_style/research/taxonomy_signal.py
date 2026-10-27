"""Training-only audit of actual paired LLM annotations and structural signal."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from .claims import sha,sentences,tokens
from .signal import overlap,distribution
from fama_style.rhetoric.concise_plan import TRAIN_SHA

ROOT=Path(__file__).resolve().parents[3]
BASE=Path('research_runs/taxonomy_discovery_v1')
LEVELS=('paragraph','sentence','language')


def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_json(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def stats(values):return distribution(values) if values else None


def checked_manifest_file(path,manifest):
    """Check only the consumed artifact, not mutable OS metadata in old manifests."""
    record=json.loads(manifest.read_text())
    if record['files'].get(path.name)!=file_sha(path):raise ValueError('Audited artifact changed')


def observation_difference(left,right):
    """Positive-label observations; missing labels do not establish absence."""
    result={}
    for level in LEVELS:
        a={r['canonical_id'] for r in left if r['level']==level and r['canonical_id']}
        b={r['canonical_id'] for r in right if r['level']==level and r['canonical_id']}
        result[level]={'source_observed':sorted(a),'target_observed':sorted(b),
            'source_only_observed':sorted(a-b),'target_only_observed':sorted(b-a),
            'jaccard':len(a&b)/len(a|b) if a|b else None,
            'absence_verified':False}
    return result


def structural_record(source,target):
    a,b=sentences(source),sentences(target)
    return {'source_sentence_count':len(a),'target_sentence_count':len(b),'sentence_delta':len(b)-len(a),
        'source_words':len(tokens(source)),'target_words':len(tokens(target)),
        'source_words_per_sentence':len(tokens(source))/max(1,len(a)),
        'target_words_per_sentence':len(tokens(target))/max(1,len(b)),
        'lexical_similarity':overlap(source,target)}


def probe_counts(rows):
    """Leave-one-training-paper-out count probe, not generation or a saved mapper."""
    if not rows or any(r['split']!='train' for r in rows):raise ValueError('Training-only probes required')
    papers=sorted({r['paper_key'] for r in rows})
    if len(papers)<2:raise ValueError('At least two training papers required')
    def sig(r):return (r['source_sentence_count'],0 if r['source_words']<=50 else 1 if r['source_words']<=100 else 2 if r['source_words']<=200 else 3)
    def modal(counter):return min(counter,key=lambda k:(-counter[k],k))
    results=[]
    for paper in papers:
        fit=[r for r in rows if r['paper_key']!=paper];held=[r for r in rows if r['paper_key']==paper]
        prior=Counter(r['target_sentence_count'] for r in fit)
        for r in held:
            conditional=Counter(x['target_sentence_count'] for x in fit if sig(x)==sig(r))
            # Five prior-equivalent observations, fixed before inspecting this run.
            pred=min(prior,key=lambda y:(-(conditional[y]+5*prior[y]/len(fit)),y))
            results.append({'paper':paper,'truth':r['target_sentence_count'],'source':r['source_sentence_count'],
                'identity':r['source_sentence_count'],'fixed_prior':modal(prior),'conditional':pred})
    methods={}
    sign=lambda value:(value>0)-(value<0)
    for name in ('identity','fixed_prior','conditional'):
        per_paper={p:statistics.mean(r[name]==r['truth'] for r in results if r['paper']==p) for p in papers}
        methods[name]={'exact_count_accuracy':statistics.mean(r[name]==r['truth'] for r in results),
            'macro_paper_exact_count_accuracy':statistics.mean(per_paper.values()),'per_paper_exact_count_accuracy':per_paper,
            'change_direction_accuracy':statistics.mean(sign(r[name]-r['source'])==sign(r['truth']-r['source']) for r in results)}
    return {'rows':len(rows),'papers':len(papers),'methods':methods,'deployment_mapper_saved':False,
        'scope':'Exploratory prediction of mechanical sentence counts; no validation/test use, semantic or author-style accuracy.'}


def summarize_group(rows):
    return {'pairs':len(rows),'papers':len({r['paper_key'] for r in rows}),
        'sentence_count_directions':dict(Counter('fewer' if r['sentence_delta']<0 else 'more' if r['sentence_delta']>0 else 'same' for r in rows)),
        'lexical_similarity':stats([r['lexical_similarity'] for r in rows]),
        'per_level_observed_set_jaccard':{level:stats([r['observed_categories'][level]['jaccard'] for r in rows if r['observed_categories'][level]['jaccard'] is not None]) for level in LEVELS},
        'annotation_sentence_count_disagreements':sum(r['annotation_sentence_counts']['source']!=r['source_sentence_count'] or r['annotation_sentence_counts']['target']!=r['target_sentence_count'] for r in rows),
        'annotation_partition_count_directions':dict(Counter('fewer' if r['annotation_sentence_counts']['target']<r['annotation_sentence_counts']['source'] else 'more' if r['annotation_sentence_counts']['target']>r['annotation_sentence_counts']['source'] else 'same' for r in rows)),
        'main_claim_position_observation_changes':sum(r['source_main_claim_position']!=r['target_main_claim_position'] for r in rows),
        'semantic_faithfulness_verified':False,'observed_set_differences_are_verified_transformations':False}


def grouping_candidates(pairs):
    """Screen count contrasts only; reject parser/annotation boundary disagreements."""
    candidates=[]
    for paper in sorted({r['paper_key'] for r in pairs}):
        group=[r for r in pairs if r['paper_key']==paper and r['sentence_delta']<0
            and r['annotation_sentence_counts']=={'source':r['source_sentence_count'],'target':r['target_sentence_count']}]
        if group:
            chosen=min(group,key=lambda r:(r['sentence_delta'],-r['lexical_similarity'],r['id']))
            candidates.append({**chosen,'operation_hypothesis':'combine_existing_sentences',
                'selection_basis':'Largest sentence-count reduction per paper after boundary-agreement screen; lexical-overlap tie break. Not semantic validation.',
                'claim_binding_status':'unresolved_cross_text_atomic_claim_alignment','training_ready':False,'execution_ready':False})
    return candidates


def validate_grouping(source,target,groups,source_sha256,target_sha256):
    """Validate source-sentence coverage, not cross-text atomic-claim equivalence."""
    if sha(source)!=source_sha256 or sha(target)!=target_sha256:raise ValueError('Grouping text changed')
    source_units=sentences(source);target_units=sentences(target)
    if len(groups)!=len(target_units) or any(not group for group in groups):raise ValueError('Wrong grouping size')
    flat=[i for group in groups for i in group]
    if any(type(i) is not int for i in flat) or flat!=list(range(1,len(source_units)+1)):
        raise ValueError('Groups must cover each source sentence once in original order')
    return {'source_sha256':source_sha256,'target_sha256':target_sha256,'groups':groups,
        'source_units':[{'unit_id':i,'text':s} for i,s in enumerate(source_units,1)],
        'target_sentence_count':len(target_units),'coverage_checked':True,
        'semantic_equivalence_verified':False,'atomic_claim_alignment_verified':False,
        'new_semantic_relations_authorized':False,'interpretation':'Assistant-proposed grouping hypothesis; no reference wording belongs in a generator plan.'}


def analyze(output):
    if output.exists():raise ValueError('Use a fresh output directory')
    inputs={}
    def read(rel):
        p=ROOT/rel;inputs[str(rel)]=file_sha(p)
        return lines(p) if p.suffix=='.jsonl' else json.loads(p.read_text())
    def audited(rel,folder):
        p=ROOT/rel;m=ROOT/BASE/folder/'manifest.json';checked_manifest_file(p,m)
        inputs[str(m.relative_to(ROOT))]=file_sha(m);return read(rel)
    training_path=Path('data/fama_corpus_v1/train.jsonl')
    if file_sha(ROOT/training_path)!=TRAIN_SHA:raise ValueError('Frozen training corpus changed')
    train=read(training_path)
    if len(train)!=247 or any(r['split']!='train' for r in train):raise ValueError('Wrong training split')
    train_by_id={r['id']:r for r in train}
    mapping=read(BASE/'package_v1/PRIVATE_MAPPING_DO_NOT_UPLOAD.json')
    targets=audited(BASE/'response_audit_v1/original_annotations.checked.private.jsonl','response_audit_v1')
    sources=audited(BASE/'response_audit_v2/annotations_B.normalized.private.jsonl','response_audit_v2')
    proposals=audited(BASE/'response_audit_v1/neutral_proposals.checked.private.jsonl','response_audit_v1')
    occurrences=audited(BASE/'consolidation_audit_v2/canonical_occurrences.private.jsonl','consolidation_audit_v2')
    catalog=read(BASE/'responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json')
    categories={r['canonical_id']:r for r in catalog['canonical_items']}
    observed=defaultdict(list);uncertain=Counter()
    for r in occurrences:
        cid=r['canonical_id']
        if cid is None:uncertain[r['annotation_id']]+=1;continue
        observed[r['annotation_id']].append({'level':categories[cid]['level'],'canonical_id':cid})
    annotation_by_id={r['paragraph_id']:r for r in targets+sources}
    pairs=[];proposed=[];identical=[]
    def pair(row,source,target,sa,ta,kind):
        # Joining uses exact underlying training IDs and texts, never annotation position.
        s_id,t_id=sa['paragraph_id'],ta['paragraph_id']
        def ann_count(a):return len(a['rhetorical_taxonomy']['sentence_structure'])
        for text,a in ((source,sa),(target,ta)):
            reconstructed=' '.join(u['sentence_text'] for u in a['rhetorical_taxonomy']['sentence_structure'])
            if ' '.join(text.split())!=' '.join(reconstructed.split()):raise ValueError('Sentence partition does not reconstruct text')
        return {'id':row['id'],'paper_key':row['paper_key'],'split':'train','cohort':kind,
            'source':source,'target':target,'source_sha256':sha(source),'target_sha256':sha(target),
            'source_annotation_id':s_id,'target_annotation_id':t_id,
            'observed_categories':observation_difference(observed[s_id],observed[t_id]),
            'unallocated_annotation_occurrences':{'source':uncertain[s_id],'target':uncertain[t_id]},
            'source_main_claim_position':sa['rhetorical_taxonomy']['paragraph_structure']['main_claim_position'],
            'target_main_claim_position':ta['rhetorical_taxonomy']['paragraph_structure']['main_claim_position'],
            'annotation_sentence_counts':{'source':ann_count(sa),'target':ann_count(ta)},
            'local_claim_ids_cross_text_aligned':False,'semantic_faithfulness_verified':False,**structural_record(source,target)}
    proposals_by_pair={r['pair_id']:r for r in proposals}
    for link in mapping:
        row=train_by_id[link['training_id']]
        if link['split']!='train' or sha(row['completion'])!=link['original_sha256'] or sha(row['plain_english'])!=link['reference_sha256']:raise ValueError('Discovery/training text mismatch')
        target=annotation_by_id[link['original_item_id']];source=annotation_by_id[link['reference_id']]
        pairs.append(pair(row,row['plain_english'],row['completion'],source,target,'discovery_existing_source'))
        proposal=proposals_by_pair[link['pair_id']];pa=proposal['proposed_neutral_annotation'];text=proposal['proposed_neutral_text']
        proposed.append({**pair(row,text,row['completion'],pa,target,'discovery_proposed_source'),'decision':proposal['decision'],
            'matches_current_training_source':text==row['plain_english'],'repairs_applied_to_corpus':False})
        if text==row['plain_english']:identical.append(pair(row,text,text,source,pa,'same_text_different_annotation_context'))
    fresh_records=audited(BASE/'stage2_audit_v1/canonical_label_records.private.jsonl','stage2_audit_v1')
    links=read(BASE/'followup_v1/PRIVATE_SELECTION_AND_LINKS.json')['fresh_annotations']
    fresh_annotations=[]
    for group in ('A','B','C'):
        # Response identities are checked against the audited stage-2 manifest.
        rel=BASE/f'responses/task_5_fresh_validation/canonical_annotations_{group}.jsonl';p=ROOT/rel
        manifest=read(BASE/'stage2_audit_v1/manifest.json')
        if manifest['input_response_hashes'][str(rel)]!=file_sha(p):raise ValueError('Fresh response changed')
        fresh_annotations+=read(rel)
    fresh_by_id={r['paragraph_id']:r for r in fresh_annotations}
    for r in fresh_records:
        if r['status']=='assigned' and r['canonical_id']:observed[r['annotation_id']].append({'level':r['level'],'canonical_id':r['canonical_id']})
        else:uncertain[r['annotation_id']]+=1
    primary=defaultdict(dict)
    for link in links:
        row=train_by_id[link['training_id']]
        if link['repeat_of']:continue
        matches=[name for name in ('plain_english','completion') if sha(row[name])==link['text_sha256']]
        if len(matches)!=1:raise ValueError('Fresh text identity ambiguous or changed')
        primary[row['id']][matches[0]]=link['paragraph_id']
    for ident,parts in primary.items():
        row=train_by_id[ident]
        if set(parts)!=set(('plain_english','completion')):raise ValueError('Unpaired fresh annotation')
        pairs.append(pair(row,row['plain_english'],row['completion'],fresh_by_id[parts['plain_english']],fresh_by_id[parts['completion']],'fresh_existing_source'))
    if len({r['id'] for r in pairs})!=len(pairs):raise ValueError('Repeated underlying pair')
    for link in links:
        if not link['repeat_of']:continue
        original=next(x for x in links if x['paragraph_id']==link['repeat_of']);row=train_by_id[link['training_id']]
        text=row['plain_english'] if sha(row['plain_english'])==link['text_sha256'] else row['completion']
        if sha(text)!=link['text_sha256'] or link['text_sha256']!=original['text_sha256']:raise ValueError('Repeat text changed')
        identical.append(pair(row,text,text,fresh_by_id[original['paragraph_id']],fresh_by_id[link['paragraph_id']],'fresh_masked_repeat'))
    corpus=[{'id':r['id'],'paper_key':r['paper_key'],'split':'train',**structural_record(r['plain_english'],r['completion'])} for r in train]
    fixed=audited(BASE/'fixed_unit_audit_final/audit_summary.json','fixed_unit_audit_final')
    candidates=grouping_candidates(pairs)
    groups={'existing_discovery_28':summarize_group([r for r in pairs if r['cohort']=='discovery_existing_source']),
        'fresh_7':summarize_group([r for r in pairs if r['cohort']=='fresh_existing_source']),
        'proposed_sources_28_not_applied':summarize_group(proposed),
        'identical_text_controls_13':summarize_group([r for r in identical if r['cohort']=='same_text_different_annotation_context']),
        'fresh_masked_repeat_controls_2':summarize_group([r for r in identical if r['cohort']=='fresh_masked_repeat'])}
    summary={'protocol':'taxonomy_training_signal_v1','execution_kind':'cpu_saved_annotation_audit','training_pairs':247,'paired_llm_taxonomy_training_pairs':len(pairs),
        'legacy_target_taxonomy_records':sum(bool(r.get('rhetorical_taxonomy')) for r in train),
        'training_pairs_without_these_paired_llm_annotations':247-len(pairs),'catalogue_categories':len(categories),'positive_labels_are_not_exhaustive_binary_features':True,
        'unallocated_discovery_occurrences':sum(r['canonical_id'] is None for r in occurrences),
        'proposed_source_decisions':dict(Counter(r['decision'] for r in proposed)),'proposed_source_texts_matching_current_corpus':sum(r['matches_current_training_source'] for r in proposed),'proposed_revisions_applied':0,
        'cohorts':groups,'corpus_sentence_count_directions':dict(Counter('fewer' if r['sentence_delta']<0 else 'more' if r['sentence_delta']>0 else 'same' for r in corpus)),
        'sentence_count_probe':probe_counts(corpus),'fixed_unit_check':{'texts':len(fixed['per_text']),'aggregate':fixed['aggregate'],'disagreements':fixed['remaining_feature_disagreements'],'limitation':'Two same-input consistency cases do not validate all 132 categories or source-to-target claim mapping.'},
        'ranked_grouping_hypotheses':len(candidates),'ready_for_full_taxonomy_mapper':False,'semantic_accuracy':None,'author_style_accuracy':None,
        'new_llm_calls':0,'gpu_jobs_submitted':0,'reserved_test_rows_read':0,'development_validation_rows_read':0,'source_data_modified':False,
        'limitations':['Discovery subset was deliberately diversified, not a population sample.','Source/proposed/target annotations have different contexts; category-set differences include annotation variability.','Positive-label omissions are not verified absence. Unallocated split mappings are preserved.','Corpus-wide counts use the legacy heuristic splitter, which can miss parenthesized sentence boundaries. Exact-text annotation partitions are reported separately for the annotated subset.','Sentence counts and lexical overlap are mechanical proxies, not semantic equivalence or distinctive style.','No new independent annotation, NLI inference or deployable mapper was produced.']}
    assert all(file_sha(ROOT/p)==h for p,h in inputs.items())
    output.mkdir(parents=True)
    for name,values in [('paired_actual.private.jsonl',pairs),('paired_proposed.private.jsonl',proposed),('identical_text_controls.private.jsonl',identical),('grouping_candidates.private.jsonl',candidates)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':summary['protocol'],'input_hashes':inputs,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'paired_llm_examples':len(pairs),'proposed_revisions_not_applied':sum(r['decision']=='revise' for r in proposed),'grouping_candidates':len(candidates),'gpu_jobs_submitted':0}));return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv);analyze(a.output_dir)


if __name__=='__main__':main()
