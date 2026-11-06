"""Build a small training-side QA recovery packet; never overwrite prior data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.research.author_curation import order_ocr_fragments, repair


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def punctuation_patch(text,before,after):
    if text.count(before)!=1 or ''.join(c for c in before if c.isalnum())!=''.join(c for c in after if c.isalnum()):
        raise ValueError('ambiguous_or_substantive_punctuation_patch')
    return text.replace(before,after,1)


def build(plan_path, output):
    if output.exists(): raise ValueError('fresh_output_required')
    plan=json.loads(plan_path.read_text())
    audit_path=ROOT/'research_runs/author_pair_expansion_v1/response_audit_v1/audits.private.jsonl'
    rows={r['paragraph_id']:r for line in audit_path.read_text().splitlines() if (r:=json.loads(line))}
    inventory_path=ROOT/'docs/author_inventory_v3/inventory.json'
    inventory={r['paper_id']:r for r in json.loads(inventory_path.read_text())['papers']}
    source_paths=['data/author_style_corpus_v1/all_blocks.private.jsonl',
                  'data/author_style_corpus_v2_1/all_blocks.private.jsonl']
    sources={};blocks={}
    for relative in source_paths:
        path=ROOT/relative;sources[relative]=sha(path)
        for line in path.read_text().splitlines():
            row=json.loads(line);blocks[row['paragraph_id']]=row
    lexicon={w.strip().lower() for w in Path('/usr/share/dict/words').read_text().splitlines()}
    inputs=[];patches=[]
    for case in plan['cases']:
        record=rows[case['paragraph_id']]
        if record['role']!='training': raise ValueError('recovery_development_or_reserved_forbidden')
        annotation=record['annotation'];target=annotation['original_paragraph'];original=target
        block=blocks[record['paragraph_id']];paper=inventory[record['paper_id']]
        if case.get('checked_page_image'):
            image=ROOT/case['checked_page_image'];sources[str(image.relative_to(ROOT))]=sha(image)
        if case['action']!='none':
            if case['action']=='punctuation_verified_on_pdf':
                before,after=case['before'],case['after']
                target=punctuation_patch(target,before,after)
            elif case['action']=='reorder_ocr_fragments':
                refs=block['line_refs'];pages={r['pdf_page'] for r in refs}
                if len(pages)!=1: raise ValueError('bounded_case_requires_one_page')
                page=None
                for relative in plan['ocr_page_files']:
                    path=ROOT/relative;sources[relative]=sha(path)
                    for line in path.read_text().splitlines():
                        candidate=json.loads(line)
                        if candidate['paper_id']==record['paper_id'] and candidate['pdf_page']==next(iter(pages)):
                            if page is not None: raise ValueError('duplicate_source_page')
                            page=candidate
                if page is None: raise ValueError('source_page_missing')
                by_id={l['line_id']:l for l in page['lines']}
                lines=[by_id[r['line_id']] for r in refs]
                reconstructed=repair('\n'.join(l['text'].strip() for l in lines),lexicon)[0]
                if reconstructed!=original: raise ValueError('stale_parent_reconstruction')
                ordered,events=order_ocr_fragments(lines)
                target=repair('\n'.join(l['text'].strip() for l in ordered),lexicon)[0]
                if not events or Counter(''.join(target.split()))!=Counter(''.join(original.split())):
                    raise ValueError('character_change_in_geometry_only_repair')
                case=dict(case,layout_events=events)
            else: raise ValueError('unknown_repair')
            patches.append(dict(case,parent_target_sha256=hashlib.sha256(original.encode()).hexdigest(),
                corrected_target_sha256=hashlib.sha256(target.encode()).hexdigest(),
                before=original,after=target,source_pdf_sha256=paper['pdf_sha256'],
                reviewer='Codex visual source check; not full-paper transcription verification'))
        siblings=sorted([b for b in blocks.values() if b['paper_id']==record['paper_id']],
                        key=lambda b:(b['pdf_page_start'],b['column_start'],b['line_refs'][0]['line_id']))
        position=next(i for i,b in enumerate(siblings) if b['paragraph_id']==record['paragraph_id'])
        context=[]
        for offset in (-1,1):
            if 0<=position+offset<len(siblings):
                sibling=siblings[position+offset]
                if sibling['section']==block['section'] and sibling['status']=='annotation_candidate':
                    context.append({'position':'preceding' if offset<0 else 'following',
                        'text':sibling['original_paragraph'],'page_location':sibling['page_location'],
                        'status':'raw_context_for_annotation_only_not_independently_verified'})
        inputs.append({'case_id':'DQ_'+record['paragraph_id'],'parent_paragraph_id':record['paragraph_id'],
                       'target_text':target,'existing_plain_english':annotation['plain_english'],
                       'previous_quality_flags':annotation['source_quality'],
                       'previous_alignment_flags':record['pair_hold_reasons'],
                       'source_check':case.get('source_check','not_visually_verified'),
                       'read_only_context':context,'target_sha256':hashlib.sha256(target.encode()).hexdigest()})
    output.mkdir(parents=True)
    common=ROOT/'docs/data_scaling_v1'
    for name in ['QA_RECOVERY_PROMPT.md','qa_recovery_output.schema.json']:
        shutil.copyfile(common/name,output/name);sources[str((common/name).relative_to(ROOT))]=sha(common/name)
    index=[]
    for start in range(0,len(inputs),6):
        n=start//6+1;name=f'inputs_{n:02d}.jsonl'
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs[start:start+6]))
        index.append({'input':name,'rows':len(inputs[start:start+6]),'response':f'responses/responses_{n:02d}.jsonl'})
    (output/'responses').mkdir()
    (output/'repairs.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in patches))
    for name,value in [('BATCH_INDEX.json',index),('session_notes.json',{'requested_model':'GPT-6','requested_effort':'High','actual_visible_settings':None}),
                       ('summary.json',{'cases':len(inputs),'batches':len(index),'source_repairs':len(patches),
                                        'status':'awaiting_targeted_QA_not_admitted','model_calls':0,'GPU_jobs':0})]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'START_HERE.md').write_text('# Targeted QA recovery\n\nUpload inputs_01.jsonl with QA_RECOVERY_PROMPT.md and qa_recovery_output.schema.json;\nrepeat for inputs_02.jsonl. Use GPT-6, request High thinking effort, and save\nresponses in responses/ using the matching batch number. Do not redo the 54 batches.\n\nThese twelve cases are training-side diagnostic selections, not a representative\nquality-rate sample. Two targets have source-checked repairs recorded privately;\nold data and raw annotations remain unchanged. Do not start training.\n')
    sources[str(plan_path.resolve().relative_to(ROOT))]=sha(plan_path)
    sources[str(audit_path.relative_to(ROOT))]=sha(audit_path)
    sources[str(inventory_path.relative_to(ROOT))]=sha(inventory_path)
    (output/'manifest.json').write_text(json.dumps({'input_sha256':sources,'code_sha256':sha(Path(__file__)),
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}},indent=2)+'\n')
    print(json.dumps({'cases':len(inputs),'batches':len(index),'source_repairs':len(patches),'model_calls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.plan,args.output)
