"""Complete the diagnostic after a documented, format-only brief projection.

Raw Gemini annotations remain untouched. Nonconforming relation labels are
retained as unqualified metadata and are never supplied to reconstruction.
"""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('purpose_prepare',ROOT/'scripts/data/prepare_purpose_500_diagnostic.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def project(row):
    b=row['brief'];points=b['key_points'];conversion='unchanged_list'
    if isinstance(points,dict):
        if any(not isinstance(k,str) or not isinstance(v,str) for k,v in points.items()):raise ValueError('point_types')
        points=[k+': '+v for k,v in points.items()]
        conversion='named_fields_to_subject_headed_list; values verbatim; no relation-index remapping needed because relations are excluded'
    if not isinstance(points,list) or not points or any(not isinstance(x,str) or not x.strip() for x in points):raise ValueError('points_required')
    if not isinstance(b['anchors'],dict) or not isinstance(b['topic'],str):raise ValueError('brief_shape')
    from fama_style.research.content_brief_v3 import PURPOSES
    if b['purpose']['primary'] not in PURPOSES or any(p not in PURPOSES for p in b['purpose']['secondary']):raise ValueError('purpose_labels')
    normalized=dict(row,brief=dict(b,key_points=points))
    projection=dict(id=row['id'],target_sha256=row['target_sha256'],brief_status=row['status'],brief_notes=row['notes'],
                    brief=dict(topic=b['topic'],purpose=b['purpose'],key_points=points,anchors=b['anchors']))
    audit=dict(id=row['id'],shape_conversion=conversion,content_edits=0,
        relation_metadata='raw retained, not schema-qualified, not projected',
        source_paragraph_forwarded=False,independent_semantic_verification=False)
    return normalized,projection,audit


def main():
    c,selected=m.check();by={r['content_id']:r for r in selected}
    briefs,raw=m.call('brief')
    if len(briefs)!=4 or {r['id'] for r in briefs}!=set(by):raise ValueError('brief_coverage')
    enhanced=[];projected=[];audits=[]
    for r in briefs:
        if r['target_sha256']!=by[r['id']]['target_sha256']:raise ValueError('target_binding')
        e,p,a=project(r);enhanced.append(e);projected.append(p);audits.append(a)
    m.write_rows(m.RUN/'enhanced_briefs.private.jsonl',enhanced)
    m.write_rows(m.RUN/'brief_projection_audit.private.jsonl',audits)
    prompt=(m.DOCS/'BRIEF_RECONSTRUCTION_V4.md').read_text()+'\nAPI transport: return a JSON ARRAY of the four objects instead of JSONL. No Markdown.\nINPUTS:\n'+json.dumps(projected,ensure_ascii=False)
    spec=m.job('reconstruction',prompt);path=m.RUN/'reconstruction_job.private.json'
    if path.exists() and m.read(path)!=spec:raise ValueError('reconstruction_job_changed')
    m.write(path,spec)
    outputs,raw=m.call('reconstruction')
    if len(outputs)!=4 or {r['id'] for r in outputs}!=set(by):raise ValueError('output_coverage')
    accepted=[]
    for r in outputs:
        if set(r)!={'id','target_sha256','plain_english','rewrite_status','review_notes'} or r['target_sha256']!=by[r['id']]['target_sha256']:raise ValueError('output_binding')
        if not isinstance(r['plain_english'],str) or not r['plain_english'].strip() or '\n\n' in r['plain_english']:raise ValueError('paragraph_required')
        if r['rewrite_status'] not in ['rewritten','limited_rewrite_room','source_problem'] or not isinstance(r['review_notes'],list):raise ValueError('output_status')
        accepted.append(dict(by[r['id']],source=r['plain_english'],source_sha256=m.training.sha(r['plain_english']),
            teacher=m.MODEL,actual_model_version=raw.get('model_version'),rewrite_status=r['rewrite_status'],review_notes=r['review_notes'],
            eligibility='diagnostic only; raw annotations preserved; no independent semantic certification'))
    accepted.sort(key=lambda r:[x['content_id'] for x in selected].index(r['content_id']))
    m.write_rows(m.RUN/'fresh_inputs.private.jsonl',accepted)
    m.write(m.RUN/'generation_manifest.json',dict(protocol=c['protocol'],builder_sha256=m.sha(Path(__file__)),
        files={str(p.relative_to(m.RUN)):m.sha(p) for p in sorted(m.RUN.rglob('*')) if p.is_file() and p.name not in ['state.json','generation_manifest.json']},
        model_calls=2,paid_fallback=False,training_calls=0,annotation_schema_correction='see brief_projection_audit.private.jsonl'))
    m.write(m.RUN/'state.json',dict(status='four_fresh_inputs_generated',model_calls=2,training_calls=0,paid_fallback=False))
    print('Saved all four fresh reconstructions; raw briefs unchanged; no training or detector calls.')


if __name__=='__main__':main()
