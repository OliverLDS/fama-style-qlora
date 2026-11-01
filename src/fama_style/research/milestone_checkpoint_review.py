"""Join frozen masked checkpoint judgments and descriptive metrics, offline only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from fama_style.evaluation.milestone_checkpoint_4b import CONDITIONS,sha
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def join(bundle,key,raw,judgments):
    cases={r['example_id']:r for r in bundle};mapping={(r['example_id'],r['label']):r for r in key}
    actual={(r['example_id'],r['condition']):r for r in raw};judged={r['example_id']:r for r in judgments['cases']}
    if (len(bundle)!=8 or len(cases)!=8 or len(key)!=24 or len(mapping)!=24 or len(raw)!=24 or len(actual)!=24
            or len(judgments['cases'])!=8 or set(judged)!=set(cases)
            or set(mapping)!={(rid,label) for rid in cases for label in 'ABC'}
            or set(actual)!={(rid,arm) for rid in cases for arm in CONDITIONS}):
        raise ValueError('exact_eight_cases_three_arms_required')
    output=[]
    for rid,case in cases.items():
        judgment=judged[rid];groups=judgment['ranking_best_first']
        if (any(not g for g in groups) or sorted(x for g in groups for x in g)!=list('ABC')
                or set(judgment['outputs'])!=set('ABC') or set(case['outputs'])!=set('ABC')):
            raise ValueError('complete_masked_judgments_required')
        ranks={label:i for i,g in enumerate(groups) for label in g}
        for label in 'ABC':
            k=mapping[(rid,label)];r=actual[(rid,k['condition'])];a=judgment['outputs'][label]
            if (sha(case['outputs'][label])!=k['output_sha256'] or case['outputs'][label]!=r['output']
                    or r['source']!=case['source'] or type(a['minor_semantic_shifts']) is not int
                    or a['minor_semantic_shifts']<0 or type(a['material_claim_change']) is not bool
                    or type(a['unresolved']) is not bool or not a['notes']):
                raise ValueError('text_or_assessment_binding_failed')
            for other in 'ABC':
                if case['outputs'][other]==r['output'] and ranks[label]!=ranks[other]:raise ValueError('identical_texts_must_tie')
            output.append(r|{'masked_label':label,'style_rank':ranks[label],'assessment':a,
                'eligible_within_local_budget':not a['material_claim_change'] and not a['unresolved'] and a['minor_semantic_shifts']<=1,
                'source_metrics':metrics(case['source'],r['output']),'reference_metrics':metrics(case['reference'],r['output'])})
    return output


def summarize(result):
    arms={}
    for arm in CONDITIONS:
        group=[r for r in result if r['condition']==arm]
        arms[arm]={'outputs':len(group),'material_claim_issues':sum(r['assessment']['material_claim_change'] for r in group),
            'unresolved':sum(r['assessment']['unresolved'] for r in group),
            'eligible_within_one_minor_shift_budget':sum(r['eligible_within_local_budget'] for r in group),
            'exact_source_copies':sum(r['copied_source'] for r in group),
            'median_source_word_sequence_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_word_sequence_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
            'median_source_length_ratio':statistics.median(r['source_metrics']['length_ratio'] for r in group)}
    control={r['example_id']:r for r in result if r['condition']=='base_ordinary'};paired={}
    for arm in CONDITIONS[1:]:
        style=Counter();meaning=Counter()
        for r in result:
            if r['condition']!=arm:continue
            old=control[r['example_id']];a=old['style_rank'];b=r['style_rank']
            style['better' if b<a else 'worse' if b>a else 'tie']+=1
            a=old['assessment']['material_claim_change'];b=r['assessment']['material_claim_change']
            meaning['improved' if a and not b else 'worse' if b and not a else 'same']+=1
        paired[arm]={'generic_exposition_preferences':{k:style[k] for k in ('better','tie','worse')},
            'material_issue_transitions':{k:meaning[k] for k in ('improved','same','worse')}}
    return {'arms':arms,'paired_against_base_ordinary':paired}


def build(packet,exported,judgments_path,historical_path,output):
    if output.exists():raise ValueError('refuse_overwrite_checkpoint_review')
    verify_files(packet);verify_files(exported)
    judgment=json.loads(judgments_path.read_text())
    if judgment['rubric']!='checkpoint_drift_rubric_v1':raise ValueError('wrong_rubric')
    result=join(rows(exported/'masked_quality_bundle.private.jsonl'),json.loads((exported/'masked_quality_key.private.json').read_text()),
        rows(exported/'combined_outputs.private.jsonl'),judgment)
    # Historical findings are retained; any new boundary disagreement is explicit.
    historical={(r['example_id'],sha(r['output'])):r['assessment'] for r in rows(historical_path)}
    disagreements=[];overlap=Counter()
    for r in result:
        old=historical.get((r['example_id'],sha(r['output'])))
        if old is None:continue
        overlap[r['condition']]+=1
        if old['material_claim_issue_identified']!=r['assessment']['material_claim_change']:
            disagreements.append({'example_id':r['example_id'],'condition':r['condition'],'output_sha256':sha(r['output']),
                'previous_material':old['material_claim_issue_identified'],'checkpoint_material':r['assessment']['material_claim_change'],
                'previous_notes':old['notes'],'checkpoint_notes':r['assessment']['notes']})
    summary={'protocol':'milestone_checkpoint_review_v1','evaluation_cases':8,'papers':4,'model_outputs':24,
        'new_outputs':14,'archived_outputs':10,'automated_judgments_provisional':True,'fama_style_established':False,
        'detector_results_observed':0,'milestone_success_established':False,'training_called':False,
        'historical_bound_outputs_compared':dict(overlap),'historical_material_boundary_disagreements':len(disagreements),
        'limits':['Same reviewer and known historical outputs; masking does not establish independence.',
                  'Eight development cases, not fresh or reserved tests; no statistical superiority claim.',
                  'Longer structural prompts repeat the source; structural mechanism is not isolated.',
                  'Readability preferences measure generic exposition, not distinctive author style.',
                  'Material/minor boundaries are provisional; historical assessments are not overwritten.'],
        **summarize(result)}
    if len(disagreements)==1 and disagreements[0]['condition']=='base_ordinary':
        summary['historical_boundary_sensitivity']={'base_material_issues_using_prior_boundary':summary['arms']['base_ordinary']['material_claim_issues']-1,
            'base_material_issues_using_checkpoint_boundary':summary['arms']['base_ordinary']['material_claim_issues']}
    output.mkdir(parents=True)
    write_rows(output/'per_output.private.jsonl',result);write_json(output/'historical_disagreements.private.json',disagreements)
    write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'package_manifest_sha256':file_sha(packet/'manifest.json'),
        'export_manifest_sha256':file_sha(exported/'manifest.json'),'masked_judgments_sha256':file_sha(judgments_path),
        'historical_review_sha256':file_sha(historical_path)})
    write_json(output/'manifest.json',{'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','exported','judgments','historical','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.exported,a.judgments,a.historical,a.output)))


if __name__=='__main__':main()
