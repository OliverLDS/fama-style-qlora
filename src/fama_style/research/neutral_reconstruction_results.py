"""Review paired Stage 2 outputs with descriptive metrics and masked judgments.

No rewriting, detector call, reference repair, model fitting or GPU execution.
Surface measurements are not semantic proof or author-style benchmarks.
"""
import argparse
from collections import Counter,defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

from .author_control_replication import verify_files
from .claims import sentences
from .neutral_reconstruction import write_json,write_rows
from fama_style.rhetoric.extract import extract,flatten

PROTOCOL='neutral_reconstruction_stage2_results_v1'
METHODS=('direct_rewrite','neutral_reconstruction')


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text):
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?",text.casefold(),re.UNICODE)


def numbers(text):
    return Counter(s.replace(',','') for s in re.findall(r'(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?',text))


def validate_output(row,method):
    if set(row)!={'record_id','method','status','text','reason'}:raise ValueError('generation_schema_fields')
    if row['method']!=method or method not in METHODS:raise ValueError('generation_method_mismatch')
    if not isinstance(row['record_id'],str) or not isinstance(row['reason'],str):raise ValueError('generation_field_types')
    if row['status']=='generated':
        if not isinstance(row['text'],str) or not row['text'].strip() or row['reason']!='':raise ValueError('generated_text_or_reason_invalid')
    elif row['status']=='blocked':
        if row['text'] is not None or not row['reason'].strip():raise ValueError('blocked_text_or_reason_invalid')
    else:raise ValueError('unknown_generation_status')


def metrics(reference,text):
    a,b=words(reference),words(text)
    matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
    fa,fb=flatten(extract(reference)['structure']),flatten(extract(text)['structure'])
    aa,bb=numbers(reference),numbers(text)
    count=len(sentences(text));reference_count=len(sentences(reference))
    return {'word_sequence_overlap':matcher.ratio(),'longest_shared_word_run':matcher.find_longest_match(0,len(a),0,len(b)).size,
            'generated_words_in_sequence_matches_fraction':sum(m.size for m in matcher.get_matching_blocks())/max(1,len(b)),
            'word_set_jaccard':len(set(a)&set(b))/max(1,len(set(a)|set(b))),
            'word_token_copy':a==b,'reference_words':len(a),'output_words':len(b),'length_ratio':len(b)/max(1,len(a)),
            'reference_sentence_count_proxy':reference_count,'output_sentence_count_proxy':count,
            'sentence_count_changed':count!=reference_count,'output_words_per_sentence_proxy':len(b)/max(1,count),
            'first_person_mentions':len(re.findall(r'\b(?:I|we|our|us|my)\b',text,re.I)),
            'reference_first_person_mentions':len(re.findall(r'\b(?:I|we|our|us|my)\b',reference,re.I)),
            'missing_numeric_values':sorted(set(aa)-set(bb)),'added_numeric_values':sorted(set(bb)-set(aa)),
            'numeric_mention_count_differences':dict((aa-bb)+(bb-aa)),
            'reference_surface_descriptors':fa,'output_surface_descriptors':fb,
            'changed_surface_descriptor_fields':[k for k in fa if fa[k]!=fb[k]],
            'multiple_paragraphs':bool(re.search(r'\n\s*\n',text)),
            'limits':'Word spelling/hyphenation affects overlap; sentence/cue rules are proxies; numeric repeats differ without changed quantities.'}


def aggregate(rows):
    generated=[r for r in rows if r['status']=='generated']
    metrics_rows=[r['metrics'] for r in generated]
    def median(key):return statistics.median(r[key] for r in metrics_rows) if metrics_rows else None
    return {'expected_output_slots':len(rows),'generated':len(generated),'blocked':len(rows)-len(generated),
            'median_word_sequence_overlap':median('word_sequence_overlap'),
            'median_longest_shared_word_run':median('longest_shared_word_run'),
            'median_word_set_jaccard':median('word_set_jaccard'),'median_length_ratio':median('length_ratio'),
            'median_sentence_count_proxy':median('output_sentence_count_proxy'),
            'median_words_per_sentence_proxy':median('output_words_per_sentence_proxy'),
            'exact_word_token_copies':sum(r['word_token_copy'] for r in metrics_rows),
            'sentence_count_changed':sum(r['sentence_count_changed'] for r in metrics_rows),
            'outputs_with_first_person':sum(r['first_person_mentions']>0 for r in metrics_rows),
            'outputs_with_missing_numeric_value':sum(bool(r['missing_numeric_values']) for r in metrics_rows),
            'outputs_with_added_numeric_value':sum(bool(r['added_numeric_values']) for r in metrics_rows),
            'material_errors_identified_in_automated_review':sum(r['assessment']['material_claim_error_identified'] for r in generated),
            'outputs_with_nonmaterial_review_flags':sum(bool(r['assessment']['local_review_flags']) for r in generated),
            'outputs_with_added_epistemic_metacommentary':sum(any(f['type']=='added_epistemic_metacommentary' for f in r['assessment']['local_review_flags']) for r in generated),
            'major_claim_order_changes_in_automated_review':sum(r['assessment']['major_claim_order_change'] for r in generated),
            'readable_in_automated_review':sum(r['assessment']['readable'] for r in generated),
            'outputs_with_readability_flags':sum(bool(r['assessment']['readability_flags']) for r in generated)}


def build(root,packet,review,output):
    root,packet,review=root.resolve(),packet.resolve(),review.resolve()
    if output.exists():raise ValueError('refuse_overwrite_results')
    verify_files(packet)
    index=json.loads((packet/'generation_index.json').read_text())
    refs={r['record_id']:r for r in map(json.loads,(packet/'private_key/reviewed_references.private.jsonl').read_text().splitlines())}
    masked=[json.loads(l) for l in (review/'blinded_bundle.private.jsonl').read_text().splitlines()]
    mapping=json.loads((review/'blinded_key.private.json').read_text())
    assessments=[json.loads(l) for l in (review/'masked_assessments.private.jsonl').read_text().splitlines()]
    map_by={(r['record_id'],r['label']):r['method'] for r in mapping}
    assessed={(r['record_id'],r['label']):r for r in assessments}
    masked_by={(r['record_id'],label):value for r in masked for label,value in r['outputs'].items()}
    if (len(mapping)!=len(map_by) or len(assessments)!=len(assessed) or len(masked)!=len(refs)
            or len(map_by)!=2*len(refs) or set(map_by)!=set(assessed) or set(map_by)!=set(masked_by)):
        raise ValueError('assessment_coverage_or_duplicate_mismatch')
    assessment_by_method={(rid,method):assessed[(rid,label)] for (rid,label),method in map_by.items()}
    received={};response_sha={};payload_sizes=Counter();rows=[]
    for task in index:
        path=packet/task['save_response_as'];response_sha[str(path.relative_to(root))]=file_sha(path)
        data=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ids=[r['record_id'] for r in data]
        if len(ids)!=len(set(ids)) or set(ids)!=set(task['record_ids']) or len(ids)!=task['expected_records']:
            raise ValueError('response_coverage_mismatch')
        payload_sizes[task['method']]+=len((packet/task['input_file']).read_bytes())
        for row in data:
            validate_output(row,task['method']);key=(row['record_id'],row['method'])
            if key in received or row['record_id'] not in refs:raise ValueError('duplicate_or_unknown_output')
            received[key]=row
    expected={(rid,method) for rid in refs for method in METHODS}
    if set(received)!=expected or set(assessment_by_method)!=expected:raise ValueError('paired_coverage_mismatch')
    for (rid,label),method in map_by.items():
        actual=received[(rid,method)];blind=masked_by[(rid,label)]
        if actual['status']!=blind['status'] or actual['text']!=blind['text']:raise ValueError('output_changed_since_masked_review')
    for record in masked:
        if record['original']!=refs[record['record_id']]['original_paragraph']:raise ValueError('reference_changed_since_review')
    for key,row in sorted(received.items()):
        rid,method=key;ref=refs[rid];assessment=assessment_by_method[key]
        valid_claims={c['claim_id'] for c in ref['brief']['claims']}
        if any(not set(flag['claim_ids'])<=valid_claims for flag in assessment['local_review_flags']):raise ValueError('unknown_review_claim_link')
        rows.append(row|{'original_paragraph':ref['original_paragraph'],'reference_sha256':ref['generation_original_sha256'],
                         'paper_id':ref['paper_id'],'style_group':ref['style_group'],
                         'metrics':metrics(ref['original_paragraph'],row['text']) if row['status']=='generated' else None,
                         'assessment':assessment,'training_ready':False})
    paired=[]
    by={(r['record_id'],r['method']):r for r in rows}
    for rid in sorted(refs):
        a,b=(by[(rid,method)] for method in METHODS)
        both=a['status']=='generated' and b['status']=='generated'
        paired.append({'record_id':rid,'paper_id':refs[rid]['paper_id'],'both_generated':both,
            'reconstruction_minus_direct_overlap':b['metrics']['word_sequence_overlap']-a['metrics']['word_sequence_overlap'] if both else None,
            'reconstruction_minus_direct_longest_run':b['metrics']['longest_shared_word_run']-a['metrics']['longest_shared_word_run'] if both else None})
    prepared=json.loads((packet/'summary.json').read_text())
    summary={'protocol':PROTOCOL,'status':'completed_descriptive_pilot_review','original_selected_cases':prepared['responses_received'],
        'stage1_held_cases':prepared['held_cases'],'paired_generation_cases':len(refs),'expected_output_slots':len(expected),
        'received_output_rows':len(rows),'generated_output_rows':sum(r['status']=='generated' for r in rows),
        'generation_time_blocks':sum(r['status']=='blocked' for r in rows),'response_files':len(response_sha),
        'methods':{method:aggregate([r for r in rows if r['method']==method]) for method in METHODS},
        'paired_complete_cases':sum(r['both_generated'] for r in paired),
        'reconstruction_lower_overlap_pairs':sum(r['both_generated'] and r['reconstruction_minus_direct_overlap']<0 for r in paired),
        'original_case_coverage_generated_per_method':{m:sum(r['method']==m and r['status']=='generated' for r in rows)/prepared['responses_received'] for m in METHODS},
        'payload_bytes_by_method':dict(payload_sizes),'hidden_token_counts_or_runtime_known':False,
        'source_reference_first_person_cases':sum(bool(re.search(r'\b(?:I|we|our|us|my)\b',r['original_paragraph'],re.I)) for r in refs.values()),
        'generation_user_provenance':json.loads((review/'session_provenance.private.json').read_text()),
        'assessment_provenance':json.loads((review/'assessment_provenance.private.json').read_text()),
        'semantic_gold_verified':False,'author_style_alignment_established':False,'detector_performance_measured':False,
        'training_ready':False,'training_pairs_created':0,'model_calls_by_agent':0,'detector_calls_by_agent':0,'GPU_jobs':0,
        'reserved_content_used':False,'outputs_rewritten_or_regenerated':False,
        'limitations':['Eight purposively selected paired cases; four original cases held before generation.',
                      'User-reported GPT-6/High UI execution; model snapshot, decoding and cross-chat retrieval uncontrolled.',
                      'Label-masked first text pass is not independent or fully blinded; reviewer knows study setup.',
                      'No identified material errors in this automated review is not proof of semantic equivalence.',
                      'Surface descriptors and lexical contrast do not establish author style, humanity or detector behavior.']}
    output.mkdir(parents=True);(output/'raw').mkdir()
    for path in response_sha:shutil.copyfile(root/path,output/'raw'/Path(path).name)
    write_rows(output/'per_output.private.jsonl',rows);write_rows(output/'paired.private.jsonl',paired)
    write_json(output/'summary.json',summary)
    write_json(output/'response_receipt.private.json',{'files':response_sha,'source_packet_manifest_sha256':file_sha(packet/'manifest.json')})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()},
        'code_sha256':file_sha(Path(__file__)),'raw_response_sha256':response_sha,
        'review_inputs_sha256':{p.name:file_sha(p) for p in review.iterdir() if p.is_file()}})
    if any(file_sha(root/name)!=digest for name,digest in response_sha.items()):raise ValueError('raw_output_modified')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    for key in ('packet','review','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();summary=build(args.root,args.packet,args.review,args.output)
    print(json.dumps({key:value for key,value in summary.items() if key not in ('generation_user_provenance','assessment_provenance','limitations')},indent=2))


if __name__=='__main__':main()
