"""CPU-only, evidence-linked claim-risk checks; no general semantic certification.

Finite detectors expose scoped counterexamples and abstain on paraphrases. They
supplement saved NLI rather than promote a high score to a semantic guarantee.
"""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from .claims import sentences,sha,verify

PROTOCOL='saved_claim_audit_v1'
BOUNDS=re.compile(r'\b(?P<upper>does not exceed|do not exceed|never exceeds?|never more than|no more than|at most|does not surpass|do not surpass)\b|\b(?P<lower>does not fall below|do not fall below|no less than|at least|not less than)\b',re.I)
STOP=set('a an the actual specified predefined pre defined minimum maximum acceptable among successive of in on for to is are was were be by it its this that these those and or but not does do when if under upon condition provided remains remain'.split())
FREQUENCY=re.compile(r'\b(?:each year|every year|per year|annually|annual)\b',re.I)
TENTATIVE=re.compile(r'\b(?:may|might|could|appears?|seems?|unlikely|possible|possibly|potentially)\b',re.I)
CERTAIN=re.compile(r'\b(?:invariably|always|necessarily|must|certainly|definitely)\b',re.I)
COPULA=re.compile(r'\b(?P<quantifier>most|many|some|all|every|no)\s+(?P<subject>[^,.;:!?]{1,70}?)\s+(?:are|is|were|was)\s+(?P<property>[^,.;:!?]{1,60})',re.I)
CONCEPTS={
    'predictive_validity':r'\b(?:predictive power|predictive validity|predictive accuracy)\b',
    'systematic_bias':r'\b(?:systematic biases?|systematic errors?)\b',
    'unobserved_mechanism':r'\b(?:unobserved (?:dependencies|effects|mechanisms)|causal mechanisms?)\b',
    'policy_implication':r'\b(?:policy implications?|policy recommendations?)\b',
}
NUMBER_WORDS=dict(zip('one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(1,21)))


def content(text):
    return {w[:-1] if len(w)>4 and w.endswith('s') and not w.endswith('ss') else w
            for w in re.findall(r'[a-z]+',text.lower()) if w not in STOP}


def related(left,right):
    a,b=content(left),content(right)
    return bool(a and a&b) and len(a&b)/len(a)>=.5


def bound_evidence(text):
    records=[]
    for index,unit in enumerate(sentences(text),1):
        for m in BOUNDS.finditer(unit):
            prefix=re.split(r'[;:!?]|\b(?:when|if|provided that)\b',unit[:m.start()],flags=re.I)[-1]
            left=' '.join(prefix.split()[-12:]);right=unit[m.end():m.end()+100]
            records.append({'unit_id':index,'direction':'upper' if m.group('upper') else 'lower',
                'cue':m.group(),'left_context':left,'right_context':right,'unit_text':unit})
    return records


def numeric_atoms(text):
    # Expand abbreviated year endpoints only for an explicit four-digit year.
    def years(m):
        start,end=m.groups();century=int(start)//100*100;finish=century+int(end)
        if finish<int(start):finish+=100
        return start+' to '+str(finish)
    normalized=re.sub(r'\b((?:18|19|20)\d{2})\s*[-–—]\s*(\d{2})\b',years,text)
    normalized=re.sub(r'\b('+ '|'.join(NUMBER_WORDS)+r')\b',lambda m:str(NUMBER_WORDS[m.group().lower()]),normalized,flags=re.I)
    return {format(Decimal(m.group()).normalize(),'f') for m in re.finditer(r'\b\d+(?:\.\d+)?\b',normalized)}


def claim_checks(source,output,nli=None):
    source_units=sentences(source);target_units=sentences(output);events=[];obligations=[]
    a,b=bound_evidence(source),bound_evidence(output)
    for bound in a:
        candidates=[x for x in b if related(bound['left_context'],x['left_context'])]
        reversed_=[x for x in candidates if x['direction']!=bound['direction']]
        obligations.append({'kind':'bound_direction','source':bound,
            'status':'opposite_direction_candidate' if reversed_ else 'same_direction_surface' if candidates else 'unresolved',
            'output_evidence':candidates})
        for x in reversed_:
            events.append({'kind':'scoped_bound_reversal_candidate','source_unit_id':bound['unit_id'],
                'source_evidence':bound,'output_evidence':x,
                'certainty':'finite_cue_and_context_match_not_verified_relation_alignment'})
    output_copulas=[(i,m) for i,s in enumerate(target_units,1) for m in COPULA.finditer(s)]
    for index,unit in enumerate(source_units,1):
        for m in COPULA.finditer(unit):
            # Compare the quantified subject AND its property; an occurrence of
            # "small" in a market-value share cannot discharge a firm-size claim.
            candidates=[(i,x) for i,x in output_copulas if related(m['subject'],x['subject']) and related(m['property'],x['property'])]
            exact=[(i,x) for i,x in candidates if x['quantifier'].lower()==m['quantifier'].lower()]
            status='same_scoped_surface' if exact else 'different_quantifier_candidate' if candidates else 'unresolved_paraphrase_or_omission'
            obligation={'kind':'quantified_property','source_unit_id':index,'source_evidence':m.group(),
                'subject':m['subject'],'property':m['property'],'quantifier':m['quantifier'],'status':status,
                'output_evidence':[{'unit_id':i,'text':x.group()} for i,x in candidates]}
            obligations.append(obligation)
            if not exact:events.append({'kind':'quantified_property_unresolved','source_unit_id':index,'evidence':obligation,
                'certainty':'could_be_paraphrase_not_confirmed_omission'})
        if FREQUENCY.search(unit):
            supported=[{'unit_id':i,'text':s} for i,s in enumerate(target_units,1) if FREQUENCY.search(s) and related(unit,s)]
            obligations.append({'kind':'annual_scope','source_unit_id':index,'status':'same_scoped_surface' if supported else 'unresolved_paraphrase_or_omission','output_evidence':supported})
            if not supported:events.append({'kind':'annual_scope_unresolved','source_unit_id':index,'source_evidence':unit,
                'certainty':'could_be_paraphrase_not_confirmed_omission'})
        if TENTATIVE.search(unit):
            stronger=[{'unit_id':i,'text':s} for i,s in enumerate(target_units,1)
                if CERTAIN.search(s) and not TENTATIVE.search(s) and related(unit,s)]
            if stronger:events.append({'kind':'modality_strengthening_candidate','source_unit_id':index,
                'source_evidence':unit,'output_evidence':stronger,
                'certainty':'surface_modal_and_context_match_negation_and_modal_scope_not_verified'})
    source_nums,output_nums=numeric_atoms(source),numeric_atoms(output)
    for kind,values in (('source_quantity_missing_candidate',source_nums-output_nums),('new_quantity_candidate',output_nums-source_nums)):
        for value in sorted(values):events.append({'kind':kind,'canonical_value':value,'certainty':'numeric_atom_only_roles_units_and_ranges_not_verified'})
    for concept,pattern in CONCEPTS.items():
        if re.search(pattern,source,re.I):continue
        evidence=[{'unit_id':i,'text':s} for i,s in enumerate(target_units,1) if re.search(pattern,s,re.I)]
        if evidence:events.append({'kind':'new_concept_candidate','concept':concept,'output_evidence':evidence,
            'certainty':'lexical_novelty_may_be_valid_paraphrase_not_confirmed_addition'})
    checks=(nli or {}).get('checks',[])
    unit_checks={int(r['direction'].split('_')[-1]):r['probabilities'] for r in checks if r['direction'].startswith('source_unit_')}
    units=[]
    for i,text in enumerate(source_units,1):
        p=unit_checks.get(i)
        units.append({'source_unit_id':i,'source_evidence':text,'saved_nli_probabilities':p,
            'saved_nli_status':'unavailable' if p is None else 'proxy_consistent' if p['entailment']>=.8 and p['contradiction']<=.1 else 'proxy_flagged',
            'event_indices':[j for j,event in enumerate(events) if event.get('source_unit_id')==i]})
    replay=verify(source,output);same=replay['status']=='surface_preserved'
    return {'protocol':PROTOCOL,'source_sha256':sha(source),'output_sha256':sha(output),
        'status':'surface_replay_only' if same and not events else 'risk_flagged' if events else 'unresolved',
        'surface_replay':replay,'events':events,'obligations':obligations,'source_units':units,
        'saved_nli_status':nli.get('status') if nli else 'not_available','semantic_proof':False,
        'limitations':'Finite cue detectors and sentence-level cached NLI; no atomic claim, actor/coreference, quantity-role, modality or causal completeness proof. No-events is not a pass. Copular and concept rules abstain on unsupported paraphrases.'}


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def audit_saved(pack_path,result_dir,verification_path,output):
    """Require existing audited bytes. Never access a model, network or test split."""
    if output.exists():raise ValueError('Use a fresh audit directory; existing audits are immutable')
    verification=json.loads(verification_path.read_text())
    if not verification.get('checks') or not all(verification['checks'].values()):raise ValueError('Prior verification did not pass')
    for name,digest in verification['artifact_hashes'].items():
        if file_sha(result_dir/name)!=digest:raise ValueError('Previously audited result changed')
    manifest=json.loads((result_dir/'manifest.json').read_text())
    status=json.loads((result_dir/'run_status.json').read_text())
    if status.get('complete') is not True or status.get('training_called') is not False:raise ValueError('Expected completed inference run')
    if file_sha(pack_path)!=manifest['input_pack_sha256']:raise ValueError('Input pack changed')
    pack=json.loads(pack_path.read_text());rows={r['id']:r for r in pack['rows']}
    if len(rows)!=4 or any(r['split']!='train' for r in rows.values()):raise ValueError('Expected four frozen training examples')
    metrics=load_lines(result_dir/'metrics.private.jsonl');gens=load_lines(result_dir/'generations.private.jsonl')
    if len(metrics)!=16 or len({(r['id'],r['condition']) for r in metrics})!=16:raise ValueError('Incomplete saved metrics')
    # Pure normalization/evaluation of already-generated text, no runtime imports.
    from fama_style.rhetoric.claim_constrained import jobs_for,realize
    jobs=jobs_for(pack);expected={(j['id'],j['condition']):j for j in jobs}
    generated={(r['id'],r['condition']):r for r in gens}
    if len(gens)!=16 or set(generated)!=set(expected):raise ValueError('Incomplete generations')
    results=[]
    for r in metrics:
        key=(r['id'],r['condition']);g=generated[key];row=rows[r['id']]
        if g['prompt_sha256']!=expected[key]['prompt_sha256'] or g['hit_output_limit'] or g['synthetic']:raise ValueError('Changed or incomplete generation')
        parsed=realize(g['output'],g['condition'],row['plan']['sentence_count'])
        if parsed['text']!=r['text'] or parsed['format_valid']!=r['format_valid']:raise ValueError('Saved normalization changed')
        audit=claim_checks(row['source'],r['text'],r['nli'])
        results.append({'id':r['id'],'condition':r['condition'],'format_valid':r['format_valid'],**audit})
    conditions={}
    for condition in sorted({r['condition'] for r in results}):
        group=[r for r in results if r['condition']==condition];types=Counter(event['kind'] for r in group for event in r['events'])
        conditions[condition]={'rows':len(group),'status_counts':dict(Counter(r['status'] for r in group)),
            'outputs_with_event':{kind:sum(any(x['kind']==kind for x in r['events']) for r in group) for kind in sorted(types)},
            'saved_nli_pass_with_events':sum(r['saved_nli_status']=='proxy_consistent' and bool(r['events']) for r in group),
            'invalid_format_rows_included':sum(not r['format_valid'] for r in group)}
    summary={'protocol':PROTOCOL,'rows':16,'conditions':conditions,'model_calls':0,'gpu_jobs_submitted':0,
        'held_out_rows_used':0,'all_claims_verified':False,'semantic_accuracy':None,
        'limitations':'Post-hoc diagnostic on four inspected training pairs, not an independent validation set. Risk counts are neither confirmed error counts nor precision/recall. A conservative replay is not general semantic certification.'}
    output.mkdir(parents=True)
    (output/'claims.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    inputs={str(p):file_sha(p) for p in (pack_path,verification_path,*[result_dir/n for n in verification['artifact_hashes']])}
    for name,digest in verification['artifact_hashes'].items():
        if file_sha(result_dir/name)!=digest:raise ValueError('Result changed during audit')
    if file_sha(pack_path)!=manifest['input_pack_sha256']:raise ValueError('Input pack changed during audit')
    (output/'manifest.json').write_text(json.dumps({'protocol':PROTOCOL,'input_hashes':inputs,
        'code_sha256':file_sha(Path(__file__)),'files':{n:file_sha(output/n) for n in ('claims.private.jsonl','summary.json')},
        'model_calls':0,'training_called':False},indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,required=True);parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args(argv);audit_saved(a.pack,a.results,a.verification,a.output_dir)


if __name__=='__main__':main()
