"""Prepare the exact eight held-out human targets as detector controls; no calls."""
import argparse
import importlib.util
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('remaining500',Path(__file__).with_name('prepare_corpus_500_remaining_scans.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
r=m.r


def select(config,development):
    ids=config['detector_case_ids']
    if len(ids)!=8 or len(set(ids))!=8:raise ValueError('eight_frozen_cases_required')
    by={x['example_id']:x for x in development}
    if len(by)!=len(development):raise ValueError('duplicate_development_ID')
    cards=[]
    for rid in ids:
        row=by[rid]
        if row['split']!='development' or row['input_condition']!='gemini_input':raise ValueError('held_out_reference_required')
        text=row['target']
        if r.runtime.sha(text)!=row['target_sha256']:raise ValueError('frozen_reference_changed')
        words=len(text.split())
        if words<50:raise ValueError('short_reference_do_not_pad')
        cards.append(dict(scan_id='SCAN_'+r.runtime.sha('mixed500-reference-pangram-v1|'+rid)[:16],
            example_id=rid,content_id=row['content_id'],parent_id=row['parent_id'],paper_id=row['paper_id'],
            style_group=row['style_group'],authors=row['authors'],condition='human_reference',
            text=text,text_sha256=row['target_sha256'],words_estimate=words,
            credits_estimate=math.ceil(words/100),provider_minimum_met=True))
    if len({x['text_sha256'] for x in cards})!=8:raise ValueError('duplicate_reference')
    if sum(x['credits_estimate'] for x in cards)>20:raise ValueError('daily_budget_requires_new_plan_do_not_shorten')
    return cards


def build(run,packet,output):
    if output.exists():raise ValueError('refuse_overwrite_reference_controls')
    m.verify_files(run/'review_v1');m.verify_files(run/'pangram_full_import_v1');m.verify_files(packet)
    report,_,_,_=r.verify(run,packet)
    splits,config=r.runtime.load_inputs(packet/'upload/corpus_500_pairs.jsonl',packet/'upload/corpus_500_config.json')
    cards=select(config,splits['development'])
    raw={x['example_id']:x for x in report['snapshots'][0]['generations']}
    for x in cards:
        if x['text']!=raw[x['example_id']]['reference']:raise ValueError('evaluation_reference_binding')
    output.mkdir();(output/'texts').mkdir();(output/'pdf_reports').mkdir()
    for x in cards:(output/'texts'/(x['scan_id']+'.txt')).write_text(x['text'])
    r.write_rows(output/'PRIVATE_KEY.jsonl',cards)
    cost=sum(x['credits_estimate'] for x in cards)
    ordered=sorted(cards,key=lambda x:r.runtime.sha('mixed500-reference-scan-order-v1|'+x['scan_id']))
    page=m.render(ordered,4,cost)
    replacements={
        '<h1>Mixed-500 — round 4</h1>':'<h1>Mixed-500 — eight human-reference controls</h1>',
        'Nine full texts from three preselected cases: each input, base output and fine-tuned output.':'Eight original human reference paragraphs for the same frozen cases already evaluated.',
        'Round 3 can follow on the next day.':'Continue after the normal daily refresh if necessary.',
        'Keep all results, including human labels and texts with meaning errors. These eighteen texts exclude the six already scanned.':'Keep every result, including any AI-related labels. These are the exact curated human reference transcriptions used in evaluation; no new extraction, rewriting or correction was applied. They are controls, not new model outputs.'}
    for old,new in replacements.items():
        if page.count(old)!=1:raise ValueError('copy_page_template_changed')
        page=page.replace(old,new,1)
    (output/'scan_references.html').write_text(page)
    summary=dict(protocol='corpus_500_reference_pangram_v1',status='prepared_not_scanned',texts=8,
        estimated_credits=cost,words_estimate=sum(x['words_estimate'] for x in cards),
        author_groups=dict(__import__('collections').Counter(x['style_group'] for x in cards)),papers=len({x['paper_id'] for x in cards}),
        purpose='Assess detector labels on the exact intended human targets, then compare all four conditions per content.',
        selection='All original eight detector IDs, unchanged order; no score-based exclusion',
        human_reference_texts_changed=False,actual_provider_counts_may_differ=True,
        new_detector_calls=0,new_model_calls=0,new_GPU_jobs=0)
    r.write(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('Open scan_references.html. Scan eight full reference paragraphs separately using the same default Pangram settings. Estimated total: '+str(cost)+' credits; actual provider counts may differ. Save all eight exported PDFs in pdf_reports/ beside this page. Default filenames are fine; no CSV or manual scores. Do not edit text, remove notation, add titles or combine paragraphs. Keep every result. No further training has started.\n')
    r.write(output/'manifest.json',dict(builder_sha256=r.sha(Path(__file__)),
        frozen_packet_manifest_sha256=r.sha(packet/'manifest.json'),
        full_panel_manifest_sha256=r.sha(run/'pangram_full_import_v1/manifest.json'),
        review_manifest_sha256=r.sha(run/'review_v1/manifest.json'),
        copy_template_sha256=r.sha(Path(m.__file__)),
        files={str(p.relative_to(output)):r.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');p.add_argument('--packet',type=Path,default=ROOT/'data/corpus_500_pilot_v1');a=p.parse_args()
    print(__import__('json').dumps(build(a.run,a.packet,a.run/'review_v1/pangram_reference_controls_v1'),indent=2))
