"""Prepare separate weakly supervised SFT/DPO research data; never trains."""
import collections
import re

from .claims import sha, variants, verify, risk_flags
from .signal import overlap, features, distance


def paraphrases(source, limit=4):
    """Two-hop bounded rewrites with replayable step-by-step provenance."""
    queue=[(source,[])];seen={source};outputs=[]
    for _ in range(2):
        next_queue=[]
        for text,path in queue:
            for candidate in variants(text):
                rewritten=candidate['text']
                if rewritten in seen or verify(text,rewritten)['status']!='surface_preserved':
                    continue
                seen.add(rewritten)
                step={"operation":candidate['operation'],"input_sha256":sha(text),"output_sha256":sha(rewritten),
                      "text":rewritten}
                chain=path+[step]
                outputs.append({"text":rewritten,"operations":chain,
                                "lexical_change":1-overlap(source,rewritten),"features":features(rewritten)})
                next_queue.append((rewritten,chain))
        queue=next_queue
    # Prefer a range of structural feature vectors; lexical-only micro-edits are reported.
    selected=[];feature_sets=set()
    for item in sorted(outputs,key=lambda item:-item['lexical_change']):
        signature=tuple(round(v,3) for v in item['features'].values())
        if signature not in feature_sets:
            selected.append(item);feature_sets.add(signature)
        if len(selected)>=limit:break
    return selected


def corruption(source):
    replacements={"may":"will","might":"will","some":"all","many":"all","not":""}
    for match in re.finditer(r"\b(may|might|some|many|not)\b",source,re.I):
        changed=source[:match.start()]+replacements[match.group().lower()]+source[match.end():]
        if risk_flags(source,changed):
            return {"text":changed,"operation":"deliberate_"+match.group().lower()+"_mutation",
                    "changed_span":[match.start(),match.end()]}
    return None


def prepare(rows, profile):
    if any(row['split']!='train' for row in rows):raise ValueError('Training preparation requires training rows only')
    sft,preferences,quarantine,coverage=[],[],[],[]
    for row in rows:
        source,target=row['plain_english'],row['completion']
        inherited=risk_flags(source,target)
        inputs=paraphrases(source)
        for variant in inputs:
            sft.append({"parent_id":row['id'],"parent_paper":row['paper_key'],"split":"train",
                        "input":variant['text'],"target":target,"target_sha256":sha(target),
                        "original_input_sha256":sha(source),"provenance":variant,
                        "reference_alignment_flags":inherited,"semantic_status":"surface_equivalent_to_curated_input_only",
                        "training_ready":False})
        coverage.append({"id":row['id'],"variants":len(inputs),
                         "lexically_diverse_variants":sum(item['lexical_change']>=.1 for item in inputs),
                         "reference_alignment_flags":inherited})
        negatives=[]
        # Avoid automatically treating already-clear or low-contrast input as bad.
        contrast=distance(source,profile)-distance(target,profile)
        if overlap(source,target)<.8 and contrast>.1:
            negatives.append({"category":"copying","text":source,"contrast_proxy":contrast})
        for candidate in variants(target):
            if candidate['operation'] in ('neutralize_author_voice','split_independent_clauses'):
                if verify(target,candidate['text'])['status']=='surface_preserved' and distance(candidate['text'],profile)-distance(target,profile)>.05:
                    negatives.append({"category":"generic_surface_proxy","text":candidate['text'],
                                      "operation":candidate['operation']})
        bad=corruption(source)
        if bad:negatives.append({"category":"semantic_error","text":bad['text'],"mutation":bad})
        for negative in negatives:
            item={"parent_id":row['id'],"split":"train","prompt":row.get('prompt',source),
                  "chosen":target,"rejected":negative['text'],"negative_category":negative['category'],
                  "chosen_sha256":sha(target),"original_input_sha256":sha(source),"provenance":negative,
                  "reference_alignment_flags":inherited,"training_ready":False,
                  "label_status":"weak_preference_not_independent_semantic_verification"}
            if inherited:quarantine.append(item)
            else:preferences.append(item)
    summary={"training_parent_rows":len(rows),"sft_variants_staged":len(sft),
             "parents_with_two_or_more_variants":sum(item['variants']>=2 for item in coverage),
             "parents_with_two_lexically_diverse_variants":sum(item['lexically_diverse_variants']>=2 for item in coverage),
             "weak_preferences":len(preferences),"quarantined_preferences":len(quarantine),
             "weak_preference_categories":dict(collections.Counter(row['negative_category'] for row in preferences)),
             "all_preference_categories":dict(collections.Counter(row['negative_category'] for row in preferences+quarantine)),
             "training_ready":False,
             "reason":"Bounded surface paraphrases are not a rich semantic-invariance generator; reference alignment and style proxies need automated model-based calibration before training.",
             "generic_negative_caveat":"Third-person/clause-splitting controls proxy generic surface style, not the entire space of fluent generic prose."}
    return {"sft":sft,"preferences":preferences,"quarantined_preferences":quarantine,"coverage":coverage,"summary":summary}


def plans():
    common={"status":"prepared_not_executed","gpu_submission_allowed":False,"model_id":"google/gemma-3-4b-it",
            "quantization":"NF4 FP32","microbatch_size":1,"gradient_accumulation_steps":2,
            "sequence_limit":1024,"truncate":False,"lora_rank":8,"language_qv_only":True,
            "reserved_test_use":False,"human_review_required":False}
    return {"A_semantic_invariant_sft":{**common,"hypothesis":"Diverse content-equivalent inputs reduce the shortcut of copying one neutral phrasing.",
                "target":"unchanged original training reference","maximum_paraphrases_per_parent":4,
                "sample_weight":"equal parent weight, not equal weight per generated row",
                "next_gate":"automatically verified diversity/coverage and no target leakage; three-example five-update memory probe only after later authorization"},
            "B_contrastive_dpo":{**common,"hypothesis":"Explicit faithful-style versus copy/generic/error contrasts change the generation preference, beyond target likelihood.",
                "reference_policy":"frozen pilot/SFT snapshot; never silently adapter-disabled raw base when SFT reference is intended",
                "inspected_trl_commit":"88365e546b5be84821f1a8bd9f167ef3230950e2",
                "peft_reference_plan":"Use the inspected implementation's frozen adapter named ref, or precomputed reference scores fingerprinted to the exact SFT snapshot; do not pass obsolete ref_adapter_name arguments",
                "unvalidated_compatibility_gates":["Gemma text path accepts fused_lm_head=True and returns log_probs", "ref adapter frozen while policy adapter updates", "cached reference scores equal uncached scores", "chosen and rejected jointly fit one T4 without truncation"],
                "trl_proposed_options":{"loss_type":["sigmoid"],"beta":.1,"learning_rate":5e-6,
                    "per_device_train_batch_size":1,"precompute_ref_log_probs":True,"precompute_ref_batch_size":1,
                    "max_length":None,"padding_free":False,"sync_ref_model":False},
                "compatibility_status":"current upstream API inspected; not installed or GPU validated",
                "next_gate":"cache reference log-probs against the exact frozen SFT adapter, explicit no-truncation audit; then separately authorized five-pair memory probe"}}
