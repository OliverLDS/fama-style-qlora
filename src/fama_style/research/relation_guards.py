"""Finite numeric-endpoint and association/causality guards with scoped abstention.

These guards supplement a fallible verifier; they never certify equivalence.
Only complete supported clauses and literal variable/entity matches are compared.
"""
from decimal import Decimal
import re

from .claims import sentences
from .claim_audit import NUMBER_WORDS

NUMBER=r'-?(?:\d+(?:\.\d+)?|'+ '|'.join(NUMBER_WORDS)+')'
ENTITY=r'[A-Za-z][A-Za-z0-9 -]{0,100}'
UNIT=r'(?:percent|%|units?|points?|degrees?)'
OPS={
    'greater than or equal to':'>=','higher than or equal to':'>=','at least':'>=','no less than':'>=','not less than':'>=',
    'less than or equal to':'<=','lower than or equal to':'<=','at most':'<=','no more than':'<=','not more than':'<=','no greater than':'<=',
    'greater than':'>','higher than':'>','more than':'>','above':'>',
    'less than':'<','lower than':'<','below':'<','>=':'>=','<=':'<=','≥':'>=','≤':'<=','>':'>','<':'<'}
OP_PATTERN='|'.join(re.escape(x) for x in sorted(OPS,key=len,reverse=True))
BOUND=re.compile(r'(?P<subject>'+ENTITY+r')\s+(?:is|are|must be)\s+(?P<operator>'+OP_PATTERN+r')\s*(?P<value>'+NUMBER+r')\s*(?P<unit>'+UNIT+r')?',re.I)
BOUND_POST=re.compile(r'(?P<subject>'+ENTITY+r')\s+(?:is|are|must be)\s+(?P<value>'+NUMBER+r')\s*(?P<unit>'+UNIT+r')?\s+or\s+(?P<direction>higher|lower|more|less)',re.I)
BOUND_NEG=re.compile(r'(?P<subject>'+ENTITY+r')\s+(?P<operator>does not exceed|do not exceed|must not exceed|does not fall below|do not fall below)\s+(?P<value>'+NUMBER+r')\s*(?P<unit>'+UNIT+r')?',re.I)
BOUND_RELEVANT=re.compile(r'(?:(?:'+OP_PATTERN+r'|does not exceed|do not exceed|must not exceed|does not fall below|do not fall below)\s*'+NUMBER+r'\b|\b'+NUMBER+r'\s*(?:'+UNIT+r')?\s+or\s+(?:higher|lower|more|less)\b)',re.I)
BLOCKED=set('not no never may might could can should would will all every some most many if unless when because that which only during after before under in according report reports reported claim claims hypothesize hypothesizes establishes establish'.split())
ASSOC=re.compile(r'(?P<a>'+ENTITY+r')\s+(?:(?:is|are)\s+(?:associated|correlated)\s+with|correlates?\s+with)\s+(?P<b>'+ENTITY+r')',re.I)
ASSOC_SYM=re.compile(r'(?P<a>'+ENTITY+r')\s+and\s+(?P<b>'+ENTITY+r')\s+(?:are|remain)\s+(?:associated|correlated)',re.I)
CAUSE=re.compile(r'(?P<a>'+ENTITY+r')\s+(?:causes?|leads? to|results? in)\s+(?P<b>'+ENTITY+r')',re.I)
CAUSE_PASSIVE=re.compile(r'(?P<b>'+ENTITY+r')\s+(?:(?:is|are)\s+caused by|results? from)\s+(?P<a>'+ENTITY+r')',re.I)
REL_RELEVANT=re.compile(r'\b(?:associated|correlated|correlates?|causes?|caused|leads? to|results? (?:in|from))\b',re.I)


def entity(text):
    text=re.sub(r'^the\s+','',text.strip().lower());words=text.split()
    if not words or set(words)&BLOCKED or words.count('and'):return None
    return ' '.join(words)


def value(text):
    text=text.lower();sign=-1 if text.startswith('-') else 1;body=text.lstrip('-')
    return format((Decimal(str(NUMBER_WORDS.get(body,body)))*sign).normalize(),'f')


def clauses(text):
    result=[]
    for i,sentence in enumerate(sentences(text),1):
        # Preserve sentence evidence even when a supported numeric condition is
        # isolated. Commas/qualifiers not recognized by the grammar cause abstention.
        for part in re.split(r';|\b(?:when|if|provided that)\b',sentence,flags=re.I):
            body=part.strip().rstrip('.!?').strip()
            if body:result.append({'sentence_id':i,'text':body,'sentence_evidence':sentence})
    return result


def observations(text):
    bounds=[];relations=[];unresolved=[]
    for c in clauses(text):
        body=c['text'];parsed_bound=False;parsed_relation=False
        for pattern in (BOUND,BOUND_POST,BOUND_NEG):
            m=pattern.fullmatch(body)
            if not m:continue
            subject=entity(m['subject'])
            if subject is None:continue
            if pattern is BOUND_POST:operator='>=' if m['direction'].lower() in ('higher','more') else '<='
            elif pattern is BOUND_NEG:operator='>=' if 'fall below' in m['operator'].lower() else '<='
            else:operator=OPS[m['operator'].lower()]
            unit=(m['unit'] or '').lower();unit='percent' if unit=='%' else unit.rstrip('s')
            bounds.append({**c,'subject':subject,'operator':operator,'value':value(m['value']),'unit':unit});parsed_bound=True;break
        conditional_relation=bool(re.search(r'\b(?:if|when|provided that|unless)\b',c['sentence_evidence'],re.I))
        for kind,pattern in (() if conditional_relation else (('association',ASSOC),('association',ASSOC_SYM),('cause',CAUSE_PASSIVE),('cause',CAUSE))):
            m=pattern.fullmatch(body)
            if not m:continue
            a,b=entity(m['a']),entity(m['b'])
            if a is None or b is None:continue
            relations.append({**c,'kind':kind,'a':a,'b':b});parsed_relation=True;break
        if BOUND_RELEVANT.search(body) and not parsed_bound:unresolved.append({**c,'kind':'unsupported_numeric_bound'})
        if REL_RELEVANT.search(body) and not parsed_relation:unresolved.append({**c,'kind':'unsupported_relation_scope'})
    return {'bounds':bounds,'relations':relations,'unresolved':unresolved}


def check(source,output):
    a,b=observations(source),observations(output);conflicts=[]
    if ' '.join(source.split())==' '.join(output.split()):
        return {'conflicts':[],'unresolved':[],'source_observations':a,'output_observations':b,
            'identity_bypass':True,'semantic_proof':False,'limitations':'Identical text has no transformation; this is not general semantic verification.'}
    unresolved=[{'side':side,**u} for side,obs in (('source',a),('output',b)) for u in obs['unresolved']]
    key=lambda x:(x['subject'],x['value'],x['unit'])
    for left in a['bounds']:
        right=[r for r in b['bounds'] if key(r)==key(left)]
        siblings=[r for r in a['bounds'] if key(r)==key(left)]
        if len(right)!=1 or len(siblings)!=1:
            unresolved.append({'kind':'ambiguous_or_unmatched_bound','source_evidence':left});continue
        if left['operator']!=right[0]['operator']:
            conflicts.append({'kind':'numeric_bound_relation_changed','source_evidence':left,'output_evidence':right[0],
                'scope':'Literal variable, value and unit match; operator changed. No integer-domain or cross-condition inference.'})
    for right in b['bounds']:
        if not any(key(left)==key(right) for left in a['bounds']):unresolved.append({'kind':'unmatched_output_bound','output_evidence':right})
    pair=lambda r:frozenset((r['a'],r['b']))
    source_causes={(r['a'],r['b']) for r in a['relations'] if r['kind']=='cause'}
    for left in a['relations']:
        aligned=[r for r in b['relations'] if pair(r)==pair(left)]
        if not aligned:unresolved.append({'kind':'unmatched_source_relation','source_evidence':left});continue
        if left['kind']=='association':
            for right in aligned:
                if right['kind']=='cause' and (right['a'],right['b']) not in source_causes:
                    conflicts.append({'kind':'association_strengthened_to_causation','source_evidence':left,'output_evidence':right,
                        'scope':'Literal entity pair matches. A causal claim elsewhere counts only for this directed pair.'})
    for right in b['relations']:
        if not any(pair(left)==pair(right) for left in a['relations']):unresolved.append({'kind':'unmatched_output_relation','output_evidence':right})
    return {'conflicts':conflicts,'unresolved':unresolved,'source_observations':a,'output_observations':b,
        'semantic_proof':False,'limitations':'Finite full-clause grammar and literal entity/unit/value alignment. Unsupported wording, negation, attribution, modalities and ambiguous scopes abstain; no general arithmetic or causal proof.'}
