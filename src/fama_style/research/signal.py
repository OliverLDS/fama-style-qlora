"""Training-only feature profiles and distributions, without ML dependencies."""
import collections
import difflib
import math
import re
import statistics

from .claims import sentences, tokens, risk_flags

FEATURES = ("words_per_sentence", "semicolon_rate", "author_voice_rate", "initial_if_rate", "parenthesis_rate")


def features(text):
    words = tokens(text)
    parts = sentences(text)
    n = max(1, len(words))
    return {"words_per_sentence": len(words)/max(1, len(parts)),
            "semicolon_rate": 100*text.count(';')/n,
            "author_voice_rate": 100*len(re.findall(r"\b(?:we|our)\b", text, re.I))/n,
            "initial_if_rate": sum(p.lower().startswith('if ') for p in parts)/max(1,len(parts)),
            "parenthesis_rate": 100*text.count('(')/n}


def overlap(a, b):
    return difflib.SequenceMatcher(None, tokens(a), tokens(b), autojunk=False).ratio()


def distribution(values, bins=None):
    ordered = sorted(values)
    def quantile(q):
        position = (len(ordered)-1)*q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    report = {"n": len(values), "min": ordered[0], "p10": quantile(.1), "p25": quantile(.25),
              "median": quantile(.5), "p75": quantile(.75), "p90": quantile(.9), "max": ordered[-1],
              "mean": statistics.mean(values)}
    if bins:
        report["histogram"] = [{"lower": a, "upper": b, "count": sum(a<=v<b or (i==len(bins)-2 and v==b) for v in values)}
                               for i,(a,b) in enumerate(zip(bins,bins[1:]))]
    return report


def fit_profile(rows):
    if not rows or any(row['split']!='train' for row in rows):
        raise ValueError('Style features can only be fitted on training references')
    source = [features(row['plain_english']) for row in rows]
    target = [features(row['completion']) for row in rows]
    profile = {"fit_split": "train", "training_rows": len(rows), "features": {},
               "limitations": "Surface style proxies, not an author classifier; topic/length confounds remain"}
    for key in FEATURES:
        a,b=[r[key] for r in source],[r[key] for r in target]
        profile['features'][key] = {"source_mean": statistics.mean(a), "reference_mean": statistics.mean(b),
                                    "reference_std": max(statistics.pstdev(b), .1),
                                    "mean_paired_delta": statistics.mean(y-x for x,y in zip(a,b)),
                                    "positive_pair_fraction": sum(y>x for x,y in zip(a,b))/len(a),
                                    "negative_pair_fraction": sum(y<x for x,y in zip(a,b))/len(a)}
    # Rules are eligible only when their direction follows the observed training delta.
    f=profile['features'];ops=[]
    if f['words_per_sentence']['mean_paired_delta']>.5:ops.append('join_independent_sentences')
    if f['words_per_sentence']['mean_paired_delta']<-.5:ops.append('split_independent_clauses')
    if f['initial_if_rate']['mean_paired_delta']>.01:ops.append('condition_to_front')
    if f['initial_if_rate']['mean_paired_delta']<-.01:ops.append('condition_to_end')
    if f['author_voice_rate']['mean_paired_delta']<-.05:ops.append('neutralize_author_voice')
    profile['eligible_operations']=ops
    return profile


def distance(text, profile):
    observed=features(text)
    return statistics.mean(abs(observed[key]-value['reference_mean'])/value['reference_std']
                           for key,value in profile['features'].items())


def diagnose(rows, profile):
    result=[]
    for row in rows:
        a,b=row['plain_english'],row['completion'];fa,fb=features(a),features(b)
        result.append({"id":row['id'],"paper_key":row['paper_key'],"lexical_overlap":overlap(a,b),
                       "word_length_ratio":len(tokens(b))/max(1,len(tokens(a))),
                       "feature_deltas":{key:fb[key]-fa[key] for key in FEATURES},
                       "style_vector_magnitude":statistics.mean(abs(fb[key]-fa[key])/profile['features'][key]['reference_std'] for key in FEATURES),
                       "alignment_risk_flags":risk_flags(a,b),
                       "alignment_status":"risk_screen_only_not_entailment"})
    summary={"rows":len(rows),"lexical_overlap":distribution([r['lexical_overlap'] for r in result],[0,.2,.4,.6,.8,.85,1]),
             "word_length_ratio":distribution([r['word_length_ratio'] for r in result]),
             "style_vector_magnitude":distribution([r['style_vector_magnitude'] for r in result]),
             "feature_delta_distributions":{key:distribution([r['feature_deltas'][key] for r in result]) for key in FEATURES},
             "alignment_risk_counts":dict(collections.Counter(flag for r in result for flag in r['alignment_risk_flags'])),
             "pairs_with_surface_alignment_flags":sum(bool(r['alignment_risk_flags']) for r in result),
             "exact_normalized_copies":sum(tokens(row['plain_english'])==tokens(row['completion']) for row in rows),
             "overlap_at_least_0_85":sum(r['lexical_overlap']>=.85 for r in result),
             "per_paper":{key:{"rows":len(group),"median_lexical_overlap":statistics.median(r['lexical_overlap'] for r in group),
                               "median_style_magnitude":statistics.median(r['style_vector_magnitude'] for r in group),
                               "mean_feature_deltas":{feature:statistics.mean(r['feature_deltas'][feature] for r in group) for feature in FEATURES}}
                          for key in sorted({r['paper_key'] for r in result})
                          for group in [[r for r in result if r['paper_key']==key]]},
             "semantic_alignment_caveat":"Counts/lexical features flag risk; differing surface markers can be equivalent and matching markers can hide drift."}
    return result, summary
