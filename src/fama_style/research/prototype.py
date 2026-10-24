"""Claim ledger → empirical plan → candidates → conservative automated checks."""
from .claims import ledger, variants, verify, sha
from .signal import distance, overlap


def rewrite(text, profile):
    claim_ledger=ledger(text)
    plan={"fit_split":"train","operations":profile['eligible_operations'],
          "source_distance":distance(text,profile),"reference_features":profile['features']}
    candidates=[]
    for candidate in variants(text):
        verification=verify(text,candidate['text'])
        gain=plan['source_distance']-distance(candidate['text'],profile)
        similarity=overlap(text,candidate['text'])
        admissible=(verification['status']=='surface_preserved' and candidate['operation'] in plan['operations']
                    and gain>.05 and similarity<.9)
        candidates.append({**candidate,"verification":verification,"style_gain_proxy":gain,
                           "copy_similarity":similarity,"eligible_for_selection":admissible})
    good=[candidate for candidate in candidates if candidate['eligible_for_selection']]
    selected=max(good,key=lambda item:item['style_gain_proxy']) if good else candidates[0]
    return {"source_sha256":sha(text),"claim_ledger":claim_ledger,"style_plan":plan,"candidates":candidates,
            "selected_text":selected['text'],"status":"selected_bounded_rewrite" if good else "abstained_unchanged",
            "semantic_guarantee":False,"scope":"CPU rule prototype; no LLM, no trained model, no independent entailment validation"}
