"""Evidence-backed surface ledger and conservative rewrite verification.

This is a bounded English-rule prototype, not a semantic parser or NLI model.
Unrecognized paraphrases abstain; matching protected words alone never passes.
"""
import collections
import hashlib
import re

TOKEN = re.compile(r"\d+(?:\.\d+)?%?|[^\W_]+(?:['’][^\W_]+)?|[<>=+−-]", re.UNICODE)
GUARD = re.compile(r"\b(if|unless|when|provided|assuming|because|therefore|thus|hence|not|no|without|may|might|could|would|should)\b", re.I)
PATTERNS = {
    "quantities": r"\b\d+(?:[.,]\d+)*(?:\s*%)?",
    "negation": r"\b(?:not|no|never|neither|nor|without)\b",
    "modality": r"\b(?:may|might|can|could|would|should|must|will|possible|possibly|expected|expect|expects)\b",
    "scope": r"\b(?:all|every|each|most|many|some|only|always|usually|generally|often|sometimes|exactly|at least|at most)\b",
    "condition_cues": r"\b(?:if|unless|when|provided that|assuming|as long as|conditional on)\b",
    "causality": r"\b(?:cause|causes|caused|lead to|leads to|because|therefore|thus|hence|implies|imply|increases|decreases|reduces|raises)\b",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tokens(text):
    return TOKEN.findall(text.casefold())


def sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if part.strip()]


def conditional(text):
    """Only recognize a single explicit if-condition with no nested guard."""
    body = text.rstrip(". ")
    front = re.fullmatch(r"If ([^,;]+), ([^;]+)", body, re.I)
    back = re.fullmatch(r"([^,;]+) if ([^,;]+)", body, re.I)
    if front:
        condition, proposition = front.groups()
    elif back:
        proposition, condition = back.groups()
    else:
        return None
    if re.search(r"\b(if|unless|only|because|then|otherwise)\b", condition+" "+proposition, re.I):
        return None
    return {"condition": condition.strip(), "proposition": proposition.strip()}


def ledger(text):
    claims = []
    for index, sentence in enumerate(sentences(text)):
        start = text.find(sentence, claims[-1]["evidence"]["end"] if claims else 0)
        protected = {key: [{"text": match.group(), "start": start+match.start(), "end": start+match.end()}
                           for match in re.finditer(pattern, sentence, re.I)] for key, pattern in PATTERNS.items()}
        condition = conditional(sentence)
        # Actor/predicate hypotheses retain exact evidence; unresolved syntax is explicit.
        actor = re.match(r"(.{1,100}?)\s+(?:is|are|was|were|has|have|may|might|can|could|would|should|must|will|find|finds|use|uses|argue|argues|predict|predicts|reduce|reduces|increase|increases|cause|causes)\b", sentence, re.I)
        actors = [actor.group(1)] if actor else []
        relations = []
        for cue in protected["causality"]:
            offset = cue["start"]-start
            left, right = sentence[:offset].strip(), sentence[offset+len(cue["text"]):].strip()
            direction = "right_explains_left" if cue["text"].lower() == "because" else "left_to_right_hypothesis"
            relations.append({"cue": cue["text"], "left": left, "right": right,
                              "direction": direction, "verified_parse": False})
        claims.append({"claim_id": f"c{index+1}", "actors": actors,
                       "proposition": condition["proposition"] if condition else sentence,
                       "conditions": ([{"text":condition["condition"],"parse_status":"single_if_rule"}] if condition else
                                      [{"text":sentence,"cue":cue['text'],"parse_status":"unresolved_scope"}
                                       for cue in protected['condition_cues']]),
                       **protected, "causal_relations": relations,
                       "evidence": {"text": sentence, "start": start, "end": start+len(sentence)},
                       "actor_parse_status": "heuristic" if actor else "unresolved"})
    return {"source_sha256": sha(text), "claims": claims,
            "coverage": "All sentence spans retained; atomic claims, coreference and implicit relations are not fully parsed",
            "independently_semantically_verified": False}


def invariants(text):
    return {key: collections.Counter(match.group().lower().replace(" ", "")
                                     for match in re.finditer(pattern, text, re.I))
            for key, pattern in PATTERNS.items()}


def risk_flags(source, target):
    a, b = invariants(source), invariants(target)
    return [key+"_surface_mismatch" for key in a if a[key] != b[key]]


def variants(text):
    """Reproducible candidate grammar; no target/reference text is used."""
    result = [{"operation": "identity", "text": text}]
    parts = sentences(text)
    # Do not fuse conditional/negated/modal/causal clauses or quoted material.
    for i in range(len(parts)-1):
        left, right = parts[i:i+2]
        if GUARD.search(left+" "+right) or any(c in left+right for c in ('"', '“', '”')):
            continue
        if not left.endswith('.') or re.search(r"(?:\b[A-Z]|e\.g|i\.e|et al)\.$", left):
            continue
        right=re.sub(r"^(The|This|These|Those|It|We|A|An)\b",lambda match:match.group().lower(),right)
        joined = left[:-1]+"; "+right
        result.append({"operation": "join_independent_sentences", "text": " ".join(parts[:i]+[joined]+parts[i+2:])})
        if len(result) >= 6:
            break
    for i, part in enumerate(parts):
        condition = conditional(part)
        if condition:
            c, p = condition["condition"], condition["proposition"]
            rewritten = p+" if "+c+"." if part.lower().startswith("if ") else "If "+c+", "+p+"."
            operation = "condition_to_end" if part.lower().startswith("if ") else "condition_to_front"
            result.append({"operation": operation, "text": " ".join(parts[:i]+[rewritten]+parts[i+1:])})
        if ";" in part and not GUARD.search(part) and not any(c in part for c in ('"','“','”')):
            pieces = part.split(";")
            if len(pieces) == 2 and all(piece.strip() for piece in pieces):
                result.append({"operation": "split_independent_clauses",
                               "text": " ".join(parts[:i]+[pieces[0].strip()+". "+pieces[1].strip()[0].upper()+pieces[1].strip()[1:]]+parts[i+1:])})
    # Explicit author voice transformation is one-directional: do not guess that
    # an arbitrary 'the authors' phrase refers to this paper rather than a citation.
    if re.search(r"\bWe\b", text) and not any(c in text for c in ('"','“','”')):
        result.append({"operation": "neutralize_author_voice", "text": re.sub(r"\bWe\b", "The authors", text)})
    unique = {}
    for candidate in result:
        unique.setdefault(candidate["text"], candidate)
    return list(unique.values())


def verify(source, candidate):
    """Pass only exact identity or a replayable conservative surface operation."""
    flags = risk_flags(source, candidate)
    if flags:
        return {"status": "rejected", "reasons": flags, "semantic_proof": False}
    replay = next((v for v in variants(source) if v["text"] == candidate), None)
    if replay is not None:
        return {"status": "surface_preserved", "operation": replay["operation"],
                "reasons": [], "semantic_proof": False,
                "scope": "Replayable bounded surface grammar; not full natural-language entailment"}
    return {"status": "unknown", "reasons": ["outside_verified_surface_grammar"], "semantic_proof": False}
