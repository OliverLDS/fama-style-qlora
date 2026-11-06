"""Audit all 500 GPT-6 inputs for weak contrast and prepare brief-first replacements.

No providers, ML weights, development/test data, or training jobs are used.
The numeric rules are reproducible triage, not a validated style classifier.
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1'
OUT = PARENT.parent / 'contrast_rewrite_v1'
RULES = dict(high_word_similarity=.80, high_ordinary_retention=.85,
             high_ordinary_fivegram_coverage=.60, long_ordinary_span=12,
             structure_word_similarity=.65, structure_fivegram_coverage=.30,
             clause_word_similarity=.65, clause_fivegram_coverage=.40, clause_ordinary_span=8,
             ordinary_bag_reuse=.85, bag_fivegram_coverage=.30, bag_ordinary_span=8,
             sentence_alignment_min=.55, sentence_layout_fraction=.80, batch_size=30)
FUNCTION_STARTS = set('the we our in this that these those a an for if but however moreover therefore thus first second third finally yet it as on by at to when there with from'.split())
# Domain terms are exempted from the ordinary-word diagnostic, not replaced.
TERMS = ('book-to-market', 'market-to-book', 'ordinary least squares', 'principal component',
         'fixed effects', 'standard errors', 'CRSP distribution codes', 'risk aversion',
         'random-walk model', 'random walk', 'lognormal distribution', 'efficient markets',
         'capital asset pricing model', 'certainty equivalent', 'moral hazard',
         'equity market timing', 'stock market', 'capital structure', 'dividend premium',
         'unit of account', 'units of account', 'social insurance', 'income per capita')
# Text-assessed follow-ups to the numerical screen, not independent human labels.
FOLLOWUPS = {
    'P106':'Clausal relocation lowers global sequence similarity, while retaining long ordinary phrases and the original evaluative frame.',
    'P144':'The data-description sequence and sentence frames remain close; numeral/definition edits obscure that continuity.',
    'P184':'The same five-sentence model comparison is recast mainly through local substitutions.',
    'P207':'All three explanatory sentence frames remain close despite substituted vocabulary.',
    'P219':'Splitting the accounting-measure sentence leaves the original sequence and expression largely intact.',
    'P253':'The same definition-and-period-statistics template remains; a brief can support a newly composed explanation.',
    'P287':'Splitting the activist list into two sentences does not substantially reframe the explanation.',
    'P319':'The same four-sentence argument remains with mostly local lexical changes.',
    'P364':'Technical anchors dominate the paragraph, but the surrounding three-sentence construction remains close; attempt fresh realization and retain a limited-room status if needed.',
    'P388':'Changing numbers to words and percent signs masks reuse of the seven-sentence statistical interpretation.',
    'P429':'The two sample-definition frames remain close; retaining dates does not require retaining their prose arrangement.',
    'P447':'The citation/result/citation progression is lightly rephrased; a content brief can support fresh synthesis.',
    'P456':'The original source-listing frame remains; citations can be retained within a newly composed account.',
    'P459':'The same six-sentence productivity argument is mostly edited locally.',
}
BRIEF_PROMPT = '''Prepare compact content briefs for fresh paragraph reconstruction.
The goal is a substantially different expression of the same economic content.
This is Stage 1 ONLY: do not write replacement paragraphs yet.

Read each human paragraph as information, not as a sentence template. Extract its
main points into neutral note fragments. Group related points by content; do not
make one bullet for each original sentence. Make causal, comparative, temporal and
conditional connections clear where they are important to the explanation.

Remove authorial voice, rhetorical openings, metaphors, distinctive adjectives,
connectives and argumentative sentence frames. For example, an author's framing
about unrealistic hopes can become notes about the inadequacy of an arrangement
and the evidence for that conclusion. Do not carry the word hope into the brief
merely because the author used it. Preserve the idea without preserving the frame.

Keep the central economic explanation and its supporting evidence. Retain names,
actual quantities, technical definitions and relationships needed to explain it.
Minor changes of emphasis and compression are acceptable. Do not invent data or
reverse the main economic relationship. If the supplied source has a genuine
problem, note it; do not guess. Do not restore stylistic phrasing for fidelity.

Return a downloadable UTF-8 JSONL file with the specified filename. Every object
must have exactly these fields:
id: copied supplied ID.
target_sha256: copied supplied string; do not calculate a new hash.
brief: {task: a short neutral writing task, key_points: [compact content notes],
        anchors: [necessary quantities, technical notation, names/citations]}.
status: ready or source_problem.
notes: array of brief uncertainty notes; empty if none.

Use as many concise points as the content needs, without padding. Anchors may keep
essential terminology and equations, but must not store quoted narrative sentences
or a disguised copy of the paragraph. Do not include original_paragraph, quotations,
old rewrites, author-style labels or rhetorical-taxonomy fields in the output.
Return all IDs in order. No Markdown fences inside the JSONL file.
Prerequisites are complete: proceed without asking for confirmation.
'''
RECONSTRUCT_PROMPT = '''Write a fresh, neutral economics paragraph from each content brief.
You receive content notes rather than an original paragraph. Construct the prose
from those notes; do not attempt to reproduce or infer an author's original wording.

Prioritize substantial rewriting: build natural new sentences, choose appropriate
grammatical subjects, and combine or separate points where useful. Avoid turning
each bullet into a sentence in the same order. Arrange the explanation coherently,
respecting important causal and temporal dependencies. Use ordinary clear prose
for an economically literate reader. Return one paragraph, without a heading.

Cover the main economic account and supporting evidence. Keep necessary quantities,
technical relationships and names. Minor shifts in emphasis or nuance are acceptable.
Do not invent facts or reverse the economic explanation. Do not let concern about
minor wording differences turn this into copying. Do not imitate a named author,
deliberately write stereotypical AI prose, or optimize for an AI detector.

Where terms or formulas are fixed, retain them and rebuild the surrounding prose.
Do not substitute synonyms merely to increase a word-change count. If there is
little room for meaningful reconstruction, give your best fresh formulation and
mark limited_rewrite_room with an explanation. If the brief has a genuine source
problem, mark source_problem and explain it without guessing.

Return a downloadable UTF-8 JSONL file with the specified filename, exactly these
five fields per input, in input order:
id: supplied ID.
target_sha256: supplied string, copied unchanged.
plain_english: a complete, newly composed neutral paragraph.
rewrite_status: rewritten, limited_rewrite_room, or source_problem.
review_notes: array of short notes; empty when none.

Do not add taxonomy, ledgers, original/reference wording, or model-setting claims.
No Markdown fences inside the JSONL file. All selected items are intended for use
after assessment and any necessary repair, not for selecting a tiny best subset.
Proceed without asking for prerequisite confirmation.
'''
BRIEF_SCHEMA = dict(type='object', additionalProperties=False,
    required=['id','target_sha256','brief','status','notes'], properties={
        'id':dict(type='string',pattern='^P[0-9]{3}$'),
        'target_sha256':dict(type='string',pattern='^[a-f0-9]{64}$'),
        'brief':dict(type='object',additionalProperties=False,required=['task','key_points','anchors'],properties={
            'task':dict(type='string',minLength=1),'key_points':dict(type='array',minItems=1,items=dict(type='string',minLength=1)),
            'anchors':dict(type='array',items=dict(type='string'))}),
        'status':dict(enum=['ready','source_problem']), 'notes':dict(type='array',items=dict(type='string'))})
REWRITE_SCHEMA = dict(type='object',additionalProperties=False,
    required=['id','target_sha256','plain_english','rewrite_status','review_notes'],properties={
        'id':BRIEF_SCHEMA['properties']['id'], 'target_sha256':BRIEF_SCHEMA['properties']['target_sha256'],
        'plain_english':dict(type='string',minLength=1),
        'rewrite_status':dict(enum=['rewritten','limited_rewrite_room','source_problem']),
        'review_notes':dict(type='array',items=dict(type='string'))})


def sha(text): return hashlib.sha256(text.encode()).hexdigest()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def jsonl(rows): return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)


def tokenize(text):
    return [(m.group().casefold(),m.group(),m.start(),m.end()) for m in re.finditer(r'\w+',text)]


def exempt_mask(text, tokens):
    spans = [m.span() for term in TERMS for m in re.finditer(r'\b'+re.escape(term)+r'\b',text,re.I)]
    mask = []
    for lower,raw,start,end in tokens:
        initial = start == 0 or bool(re.search(r'[.!?]\s*["“(]*$',text[:start]))
        exempt = (any(c.isdigit() for c in raw) or '_' in raw or re.search('[α-ωΑ-Ω]',raw)
                  or (raw.isupper() and len(raw)>1)
                  or (raw[0].isupper() and not initial and lower not in FUNCTION_STARTS)
                  or any(start>=a and end<=b for a,b in spans))
        mask.append(bool(exempt))
    return mask


def sentences(text):
    for a,b in [('U.S.','US'),('et al.','et al'),('e.g.','eg'),('i.e.','ie')]: text=text.replace(a,b)
    return [r for r in re.split(r'(?<=[.!?])\s+(?=[A-Z“"(])',text) if r.strip()]


def contrast(source,target):
    a=tokenize(target); b=tokenize(source); at=[r[0] for r in a];bt=[r[0] for r in b]
    if not at or not bt: raise ValueError('empty_pair')
    mask=exempt_mask(source,b);ordinary={i for i in range(len(b)) if not mask[i]}
    target_mask=exempt_mask(target,a)
    target_bag=Counter(t[0] for t,x in zip(a,target_mask) if not x)
    source_bag=Counter(b[i][0] for i in ordinary)
    matcher=SequenceMatcher(None,at,bt,autojunk=False);blocks=matcher.get_matching_blocks()
    copied={i for block in blocks for i in range(block.b,block.b+block.size)}
    fivegrams={tuple(at[i:i+5]) for i in range(len(at)-4)};fivecovered=set()
    for i in range(len(bt)-4):
        if tuple(bt[i:i+5]) in fivegrams and sum(j in ordinary for j in range(i,i+5))>=2:
            fivecovered.update(j for j in range(i,i+5) if j in ordinary)
    longest=max(blocks,key=lambda x:x.size)
    longest_ordinary=max((sum(j in ordinary for j in range(block.b,block.b+block.size)) for block in blocks),default=0)
    ss=sentences(source); ts=sentences(target); matches=[]
    for s in ss:
        st=[r[0] for r in tokenize(s)]
        scores=[SequenceMatcher(None,st,[r[0] for r in tokenize(t)],autojunk=False).ratio() for t in ts]
        j=max(range(len(scores)),key=scores.__getitem__)
        matches.append((j,scores[j]))
    ordered=[]
    for j,score in matches:
        if score>=RULES['sentence_alignment_min'] and (not ordered or j>ordered[-1]): ordered.append(j)
    layout_fraction=len(ordered)/max(len(ss),len(ts))
    same_layout=len(ss)==len(ts) and layout_fraction>=RULES['sentence_layout_fraction']
    return dict(word_similarity=matcher.ratio(),ordinary_token_count=len(ordinary),
        exempt_token_fraction=1-len(ordinary)/len(bt),
        ordinary_retention=len(copied & ordinary)/max(1,len(ordinary)),
        ordinary_bag_reuse=sum((target_bag & source_bag).values())/max(1,len(ordinary)),
        ordinary_shared_fivegram_coverage=len(fivecovered)/max(1,len(ordinary)),
        longest_copied_words=longest.size,longest_ordinary_words=longest_ordinary,
        longest_copied_span=' '.join(r[1] for r in b[longest.b:longest.b+longest.size]),
        source_sentences=len(ss),target_sentences=len(ts),ordered_sentence_alignment_fraction=layout_fraction,
        sentence_layout_echo=same_layout,word_length_ratio=len(source.split())/len(target.split()))


def classify(metrics, target_issue=False, review_id=None):
    if target_issue: return 'target_cleanup_first',['Inherited target issue; avoid wasting a rewrite request.']
    m=metrics;reasons=[]
    if m['word_similarity']>=RULES['high_word_similarity']:
        reasons.append('High overall word-sequence similarity.')
    if m['ordinary_token_count']>=15 and m['ordinary_retention']>=RULES['high_ordinary_retention']:
        reasons.append('Most ordinary wording remains in the same token alignment.')
    if (m['ordinary_shared_fivegram_coverage']>=RULES['high_ordinary_fivegram_coverage']
            and m['longest_ordinary_words']>=RULES['long_ordinary_span']):
        reasons.append('Extensive unchanged prose spans beyond obvious fixed material.')
    if (m['word_similarity']>=RULES['structure_word_similarity'] and m['sentence_layout_echo']
            and m['ordinary_shared_fivegram_coverage']>=RULES['structure_fivegram_coverage']):
        reasons.append('Substantial shared prose plus the same sentence-count/alignment pattern.')
    if (m['word_similarity']>=RULES['clause_word_similarity']
            and m['ordinary_shared_fivegram_coverage']>=RULES['clause_fivegram_coverage']
            and m['longest_ordinary_words']>=RULES['clause_ordinary_span']):
        reasons.append('Unchanged ordinary clauses remain; splitting sentences alone does not establish fresh expression.')
    if (m['ordinary_bag_reuse']>=RULES['ordinary_bag_reuse']
            and m['ordinary_shared_fivegram_coverage']>=RULES['bag_fivegram_coverage']
            and m['longest_ordinary_words']>=RULES['bag_ordinary_span']):
        reasons.append('Nearly the same ordinary-word inventory and retained phrases despite clause reordering.')
    if review_id in FOLLOWUPS: reasons.append('Text follow-up: '+FOLLOWUPS[review_id])
    return ('regenerate_brief_first' if reasons else 'retain_current'), reasons


def verify_quality_parent():
    manifest=read(PARENT/'quality_manifest.json')
    for name,h in manifest['files'].items():
        if digest(PARENT/name)!=h: raise ValueError('quality_parent_changed')
    rows=lines(PARENT/'candidates_0500.private.jsonl')
    if len(rows)!=500 or len({r['parent_target_id'] for r in rows})!=500: raise ValueError('matched_500_required')
    return rows


def interleave(records):
    pools=defaultdict(list)
    for r in records: pools[r['style_group']].append(r)
    for pool in pools.values(): pool.sort(key=lambda r:(-r['metrics']['word_similarity'],r['id']))
    priority=[r for rid in ['P002','P024'] for r in records if r['id']==rid]
    done={r['id'] for r in priority};out=list(priority)
    while any(pools.values()):
        for name in sorted(pools):
            if pools[name]:
                r=pools[name].pop(0)
                if r['id'] not in done:out.append(r);done.add(r['id'])
    return out


def write_jobs(folder, records, phase):
    folder.mkdir();jobs=[]
    for start in range(0,len(records),RULES['batch_size']):
        chunk=records[start:start+RULES['batch_size']];i=len(jobs)+1
        output=f'briefs_{i:02d}.jsonl' if phase=='briefs' else f'rewrites_{i:02d}.jsonl'
        inputs=[dict(id=r['id'],target_sha256=r['target_sha256'],original_paragraph=r['target']) for r in chunk] if phase=='briefs' else chunk
        prompt=BRIEF_PROMPT if phase=='briefs' else RECONSTRUCT_PROMPT
        (folder/f'inputs_{i:02d}.jsonl').write_text(jsonl(inputs))
        (folder/f'batch_{i:02d}.md').write_text(prompt+f'\nOutput filename: {output}\nExpected records: {len(chunk)}\n\nINPUTS:\n```jsonl\n'+jsonl(inputs)+'```\n')
        jobs.append(dict(batch=i,ids=[r['id'] for r in chunk],records=len(chunk),attachment=f'batch_{i:02d}.md',response=output))
    dump(folder/'jobs.json',jobs)
    return jobs


def prepare():
    if OUT.exists(): raise ValueError('refuse_overwrite_contrast_packet')
    rows=verify_quality_parent();issues={r['id']:r for r in lines(PARENT/'adjudications.private.jsonl') if r['decision']=='target_review_pending'}
    audit=[]
    for i,row in enumerate(rows,1):
        rid=f'P{i:03d}'
        if sha(row['source'])!=row['source_sha256'] or sha(row['target'])!=row['target_sha256'] or row['split']!='training':raise ValueError('parent_text_binding')
        metrics=contrast(row['source'],row['target']);decision,reasons=classify(metrics,rid in issues,rid)
        audit.append(dict(id=rid,mixed_id=row['provenance']['matched_mixed_id'],source_sha256=row['source_sha256'],
            target_sha256=row['target_sha256'],source=row['source'],target=row['target'],style_group=row['style_group'],
            paper_id=row['paper_id'],metrics=metrics,decision=decision,reasons=reasons,
            origin=row['provenance']['input_origin'],independent_style_verification=False))
    selected=interleave([r for r in audit if r['decision']=='regenerate_brief_first'])
    if next(r for r in audit if r['id']=='P001')['decision']!='retain_current' or not {'P002','P024'}<={r['id'] for r in selected}:raise ValueError('feedback_anchor_inconsistency')
    OUT.mkdir();(OUT/'responses').mkdir();(OUT/'responses/stage_1_briefs').mkdir();(OUT/'responses/stage_2_rewrites').mkdir()
    (OUT/'audit.private.jsonl').write_text(jsonl(audit));(OUT/'selected_targets.private.jsonl').write_text(jsonl(selected))
    dump(OUT/'rules.json',dict(protocol='gpt6_contrast_rewrite_v1',rules=RULES,exempt_terms=TERMS,text_followups=FOLLOWUPS,
        limitations=['Finite obvious-token exemptions are incomplete; ordinary means not exempted by this heuristic.',
                     'Sentence counts and alignments are surface proxies, not syntactic/discourse annotation.',
                     'Thresholds prioritize regeneration; they are not a validated quality benchmark.',
                     'No low-overlap score proves meaning or distinctive author-style quality.']))
    (OUT/'STAGE_1_BRIEF_PROMPT.md').write_text(BRIEF_PROMPT);(OUT/'STAGE_2_RECONSTRUCTION_PROMPT.md').write_text(RECONSTRUCT_PROMPT)
    dump(OUT/'brief_output.schema.json',BRIEF_SCHEMA);dump(OUT/'rewrite_output.schema.json',REWRITE_SCHEMA)
    jobs=write_jobs(OUT/'stage_1_brief_batches',selected,'briefs')
    summary=dict(protocol='gpt6_contrast_rewrite_v1',reviewed_training_pairs=500,
        decisions=dict(Counter(r['decision'] for r in audit)),regeneration_pairs=len(selected),stage_1_batches=len(jobs),
        batch_sizes=[j['records'] for j in jobs],selected_by_group=dict(Counter(r['style_group'] for r in selected)),
        selected_by_origin=dict(Counter(r['origin'] for r in selected)),
        metrics_medians={k:statistics.median(r['metrics'][k] for r in audit) for k in ['word_similarity','ordinary_retention','ordinary_shared_fivegram_coverage']},
        original_targets_changed=0,current_sources_changed=0,provider_calls=0,training_calls=0,
        development_examples_read=0,reserved_test_examples_read=0,
        stage_2_inputs_prepared=False,stage_2_waits_for='The actual Stage 1 brief responses; reconstruction never receives original/reference prose.')
    dump(OUT/'summary.json',summary)
    (OUT/'START_HERE.md').write_text(f'''# GPT-6 contrast repair: brief first, then reconstruct

All 500 current GPT-6 inputs were contrast-screened. {len(selected)} are selected
for replacement inputs; all human targets and old sources stay unchanged.
There are {len(jobs)} Stage 1 uploads. Every selected target is intended for completion,
not for selecting a small best subset. Five inherited target problems are deferred
instead of spending uploads on unsuitable/incomplete source text.

## Stage 1 — neutral content briefs

Use GPT-6 and request High thinking effort. Upload only
stage_1_brief_batches/batch_01.md, and ask it to complete the attached job and return
the named downloadable JSONL file. Repeat for the other batch files. Each file
contains its own instructions and human paragraphs; no extra attachments are needed.
Save briefs_01.jsonl etc. in responses/stage_1_briefs/.

These are content notes, not paragraph rewrites or rhetorical-taxonomy annotations.
Do not upload earlier weak rewrites as examples. P002 and P024 are in the first job.

## Stage 2 — fresh paragraphs from briefs only

When the brief responses are ready, tell Codex, or run:
python3 scripts/data/prepare_gpt6_contrast_rewrites.py prepare-reconstruction

The CPU builder checks ID/hash coverage and prepares stage_2_reconstruction_batches/.
Then upload each Stage 2 batch in a separate request using GPT-6/High. Prefer a new
chat without attaching the original paragraphs, old rewrites, or Stage 1 attachments.
Save rewrites_01.jsonl etc. in responses/stage_2_rewrites/.

Stage 2 receives only neutral briefs, IDs and hash strings. No claim of isolation
from ChatGPT cross-chat memory is made. Do not manually paste the original alongside
the brief; that would recreate the close-paraphrase setup.

## Quality and preservation

The priority is fresh wording and sentence construction. Main economic content and
factual anchors matter, while modest emphasis shifts are acceptable. Reviews must
not restore expressive author wording just to make the neutral input closer.
Word-change percentages are diagnostics, not a substitute for useful reconstruction.
After responses return, assess contrast and material errors separately. There is no
automatic admission/rejection based on a detector score or one overlap threshold.

No current pair was deleted, replaced or released for training. No API/GPU calls or
manual detector scans are requested. The five deferred target issues affect both
teacher tracks and still require a shared versioned cleanup. Gemini held-out inputs
remain a separate evaluation option for a GPT-6-trained student; compare input and
output detector scores on paper-held-out cases, without selecting on detector success.

The audit is automated lexical/span/sentence-layout triage across all 500 training
pairs, not independent human certification of 500 style judgments. See rules.json
and audit.private.jsonl for each decision and its evidence.
''')
    dump(OUT/'manifest.json',dict(parent_quality_manifest_sha256=digest(PARENT/'quality_manifest.json'),
        parent_candidates_sha256=digest(PARENT/'candidates_0500.private.jsonl'),builder_sha256=digest(Path(__file__)),
        files={str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()}))
    print(json.dumps(summary,indent=2))


def check():
    m=read(OUT/'manifest.json')
    if digest(PARENT/'quality_manifest.json')!=m['parent_quality_manifest_sha256'] or digest(PARENT/'candidates_0500.private.jsonl')!=m['parent_candidates_sha256'] or digest(Path(__file__))!=m['builder_sha256']:raise ValueError('changed_dependency')
    for name,h in m['files'].items():
        if digest(OUT/name)!=h:raise ValueError('changed_packet')
    return m


def validate_brief(row, expected):
    if not isinstance(row,dict) or set(row)!=set(BRIEF_SCHEMA['required']):raise ValueError('brief_fields')
    if row['id']!=expected['id'] or row['target_sha256']!=expected['target_sha256']:raise ValueError('brief_identity')
    brief=row['brief']
    if not isinstance(brief,dict) or set(brief)!={'task','key_points','anchors'}:raise ValueError('brief_structure')
    if not isinstance(brief['task'],str) or not brief['task'].strip() or not isinstance(brief['key_points'],list) or not brief['key_points']:raise ValueError('empty_brief')
    for key in ['key_points','anchors']:
        if not isinstance(brief[key],list) or any(not isinstance(x,str) or not x.strip() for x in brief[key]):raise ValueError('brief_list')
    if row['status'] not in {'ready','source_problem'} or not isinstance(row['notes'],list) or any(not isinstance(n,str) for n in row['notes']):raise ValueError('brief_status')
    if row['status']=='source_problem' and not row['notes']:raise ValueError('unexplained_source_problem')
    if any(' '.join(x.casefold().split())==' '.join(expected['target'].casefold().split()) for x in [brief['task']]+brief['key_points']+brief['anchors']):raise ValueError('whole_original_disguised_as_brief')


def prepare_reconstruction():
    check();folder=OUT/'stage_2_reconstruction_batches'
    if folder.exists():raise ValueError('refuse_overwrite_reconstruction')
    selected=lines(OUT/'selected_targets.private.jsonl');by={r['id']:r for r in selected};records=[];response_hashes={};alerts=[]
    for job in read(OUT/'stage_1_brief_batches/jobs.json'):
        name='responses/stage_1_briefs/'+job['response'];path=OUT/name;rows=lines(path)
        if len(rows)!=len(job['ids']):raise ValueError('brief_batch_coverage')
        response_hashes[name]=digest(path)
        for row,rid in zip(rows,job['ids']):
            validate_brief(row,by[rid]);brief_text=' '.join([row['brief']['task']]+row['brief']['key_points']+row['brief']['anchors'])
            c=contrast(brief_text,by[rid]['target'])
            if c['longest_ordinary_words']>=12:alerts.append(dict(id=rid,flag='long_shared_ordinary_span_in_brief',words=c['longest_ordinary_words']))
            if row['status']=='source_problem':alerts.append(dict(id=rid,flag='brief_reports_source_problem',notes=row['notes']))
            records.append(dict(id=rid,target_sha256=row['target_sha256'],brief=row['brief'],brief_status=row['status'],brief_notes=row['notes']))
    if len(records)!=len(selected) or len({r['id'] for r in records})!=len(selected):raise ValueError('brief_full_coverage')
    jobs=write_jobs(folder,records,'reconstruction')
    dump(folder/'brief_screen.json',dict(alerts=alerts,policy='Advisory checks; no brittle numeric/wording gate. Whole-original copies are rejected. Source problems remain explicit for assessment.'))
    dump(folder/'manifest.json',dict(parent_preparation_manifest_sha256=digest(OUT/'manifest.json'),response_files=response_hashes,
        records=len(records),no_original_paragraph_field=True,no_old_rewrite_field=True,provider_calls=0,training_calls=0,
        files={str(f.relative_to(folder)):digest(f) for f in folder.iterdir() if f.is_file()}))
    print(f'Prepared {len(records)} brief-only reconstructions in {len(jobs)} uploads. {len(alerts)} advisory brief flags retained; no API/training calls.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','check','prepare-reconstruction'])
    args=parser.parse_args()
    {'prepare':prepare,'check':check,'prepare-reconstruction':prepare_reconstruction}[args.action]()
