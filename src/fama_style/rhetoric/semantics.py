"""Independent NLI diagnostic interface, with automatic controls and abstention."""
import math

from fama_style.research.claims import sentences

NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


def checked_prediction(predict, premise, hypothesis):
    result = predict(premise, hypothesis)
    if result is None: return None  # Context overflow must abstain, never truncate.
    labels = {'entailment', 'neutral', 'contradiction'}
    if set(result)!=labels or any(not isinstance(p, (float,int)) or not math.isfinite(p) or not 0<=p<=1 for p in result.values()):
        raise ValueError('Invalid NLI probability mapping')
    if abs(sum(result.values())-1)>.01: raise ValueError('Invalid NLI probability sum')
    return result


def assess(source, output, predict):
    pairs = [('whole_forward', source, output), ('whole_reverse', output, source)]
    pairs += [(f'source_unit_{i}', output, text) for i,text in enumerate(sentences(source),1)]
    pairs += [(f'output_unit_{i}', source, text) for i,text in enumerate(sentences(output),1)]
    records = [{'direction': direction, 'probabilities': checked_prediction(predict, premise, hypothesis)}
               for direction, premise, hypothesis in pairs]
    known = [row['probabilities'] for row in records if row['probabilities'] is not None]
    complete = len(known)==len(records)
    minimum = min((p['entailment'] for p in known), default=None)
    maximum = max((p['contradiction'] for p in known), default=None)
    return {'status': 'unknown_over_context' if not complete else 'proxy_consistent' if minimum>=.8 and maximum<=.1 else 'proxy_flagged',
            'minimum_entailment': minimum, 'maximum_contradiction': maximum, 'checks': records,
            'semantic_proof': False, 'sentence_units_are_atomic_claims': False}


def calibration_controls(predict):
    # Authored synthetic controls; no validation or reserved-test references.
    equivalent = [('Firms report revenue.', 'Firms report revenue.'),
                  ('If demand falls, firms reduce output.', 'Firms reduce output if demand falls.'),
                  ('Firms report revenue. Investors read reports.', 'Firms report revenue; investors read reports.')]
    changed = [('Firms pay investors.', 'Investors pay firms.'),
               ('Costs cause losses.', 'Losses cause costs.'),
               ('If demand falls, firms reduce output.', 'Firms reduce output.'),
               ('Some firms invest.', 'All firms invest.'),
               ('Firms may reduce output.', 'Firms must reduce output.'),
               ('Firms do not invest.', 'Firms invest.'),
               ('Revenue rises by 5 percent.', 'Revenue rises by 10 percent.')]
    positives = [assess(a,b,predict)['status']=='proxy_consistent' for a,b in equivalent]
    negatives = [assess(a,b,predict)['status']=='proxy_flagged' for a,b in changed]
    passed = all(positives) and all(negatives)
    return {'equivalent_controls': len(positives), 'accepted_equivalents': sum(positives),
            'semantic_mutations': len(negatives), 'rejected_mutations': sum(negatives),
            'passed': passed, 'thresholds': {'entailment_minimum': .8, 'contradiction_maximum': .1},
            'meaning': 'Small synthetic sanity gate only; not domain calibration or verified economic entailment',
            'human_review_used': False}
