"""Training-only learnability probes; no mapping is deployed before oracle evidence."""
import collections
import math
import statistics

from .extract import flatten, PROTECTED_FEATURES


def entropy(values):
    counts = collections.Counter(values); n = sum(counts.values())
    return -sum(count/n * math.log2(count/n) for count in counts.values()) if n else 0.0


def mode(values):
    counts = collections.Counter(values)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def probe(pairs):
    if not pairs or any(row['split']!='train' for row in pairs):
        raise ValueError('Only training pairs may inform mapping probes')
    rows = [{'paper': row['paper_key'], 'x': flatten(row['annotation']['source']['structure']),
             'y': flatten(row['annotation']['target']['structure'])} for row in pairs]
    keys = [key for key in rows[0]['x'] if key not in PROTECTED_FEATURES]
    if len({row['paper'] for row in rows}) < 2: raise ValueError('Need independent paper folds')
    predictions = {name: [] for name in ('source_identity', 'fixed_prior', 'retrieval', 'smoothed_conditional')}
    per_paper = {}
    for paper in sorted({row['paper'] for row in rows}):
        fit = [row for row in rows if row['paper']!=paper]
        held = [row for row in rows if row['paper']==paper]
        local = {name: [] for name in predictions}
        for row in held:
            nearest = min(enumerate(fit), key=lambda item:(sum(row['x'][key]!=item[1]['x'][key] for key in keys), item[0]))[1]
            pred = {name:{} for name in predictions}
            for key in keys:
                prior = collections.Counter(item['y'][key] for item in fit)
                conditioned = collections.Counter(item['y'][key] for item in fit if item['x'][key]==row['x'][key])
                # Five prior-equivalent observations regularize rare source buckets.
                labels = sorted(prior)
                conditional = max(labels, key=lambda label: conditioned[label] + 5*prior[label]/len(fit))
                pred['source_identity'][key] = row['x'][key]
                pred['fixed_prior'][key] = mode(item['y'][key] for item in fit)
                pred['retrieval'][key] = nearest['y'][key]
                pred['smoothed_conditional'][key] = conditional
            for name in pred:
                matches = {key: pred[name][key]==row['y'][key] for key in keys}
                predictions[name].append(matches); local[name].append(matches)
        per_paper[paper] = {name: statistics.mean(sum(record.values())/len(keys) for record in records)
                            for name, records in local.items()}
    scores = {name: {'macro_field_accuracy': statistics.mean(sum(row.values())/len(keys) for row in records),
                     'per_field_accuracy': {key: statistics.mean(row[key] for row in records) for key in keys}}
              for name, records in predictions.items()}
    return {'rows': len(rows), 'papers': len(per_paper), 'fit_split': 'train',
        'source_signatures': len({tuple(row['x'][key] for key in keys) for row in rows}),
        'target_signatures': len({tuple(row['y'][key] for key in keys) for row in rows}),
        'protected_features_excluded_from_prediction': sorted(PROTECTED_FEATURES),
        'per_field': {key: {'source_counts': dict(collections.Counter(row['x'][key] for row in rows)),
            'target_counts': dict(collections.Counter(row['y'][key] for row in rows)),
            'source_entropy_bits': entropy([row['x'][key] for row in rows]),
            'target_entropy_bits': entropy([row['y'][key] for row in rows]),
            'changed_fraction': statistics.mean(row['x'][key]!=row['y'][key] for row in rows)} for key in keys},
        'leave_one_training_paper_out': scores, 'per_paper_macro_accuracy': per_paper,
        'protocol': 'Exploratory categorical probes; validation and reserved test excluded; no deployable mapping fitted',
        'recommendation': 'Test the oracle first. If supported, start with fixed prior; require paper-grouped gains before adding retrieval or smoothed conditional edits. No neural mapper yet.',
        'caveat': 'Bucket prediction accuracy is not generation quality or semantic preservation; field imbalance and correlated controls inflate accuracy.'}
