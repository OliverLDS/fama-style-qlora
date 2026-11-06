"""Purpose briefs with finer content relations; preserves the frozen v2 contract."""
import copy
from .content_brief import PURPOSES, validate_brief_v2

# label: (family, direction/meaning, v2 shape-validation equivalent).
# Equivalents support validation only, not claims of semantic interchangeability.
RELATION_SPEC = {
    'cause': ('causal', 'Cause to outcome; retain any stated uncertainty.', 'cause'),
    'contributing_cause': ('causal', 'One contributing factor to outcome; does not assert sufficiency or exclusivity.', 'cause'),
    'mechanism_step': ('causal', 'Earlier link to next link in a described transmission mechanism.', 'cause'),
    'enables': ('causal', 'Resource or arrangement to activity it makes feasible; does not assert that the activity occurs.', 'condition'),
    'prevents': ('causal', 'Intervention or constraint to the outcome it inhibits; retain partial versus complete prevention.', 'cause'),
    'feedback': ('causal', 'From one process to another that also affects the first; use only when the return link is stated.', 'cause'),
    'association': ('empirical', 'Associated variables or patterns; no causal or time direction is asserted.', 'association'),
    'predicts': ('empirical', 'Predictor to forecast outcome; statistical prediction does not imply causation.', 'association'),
    'necessary_condition': ('logical', 'Requirement to the outcome requiring it; necessary does not mean sufficient.', 'condition'),
    'sufficient_condition': ('logical', 'Sufficient premise to entailed outcome within the stated model/scope; not a necessary condition.', 'condition'),
    'conditional_consequence': ('logical', 'Stated condition to consequence within that case, without asserting a universal necessary/sufficient rule.', 'condition'),
    'logical_implication': ('logical', 'Premise or model result to the conclusion derived from it; not automatically a causal effect.', 'condition'),
    'counterfactual_dependence': ('logical', 'Hypothetical change to the hypothetical outcome; retain the counterfactual status.', 'condition'),
    'similarity': ('comparison', 'Points share a relevant property; no ranking or causal direction is asserted.', 'contrast'),
    'contrast': ('comparison', 'Points differ on an identified dimension; no causal direction is asserted.', 'contrast'),
    'quantitative_comparison': ('comparison', 'First quantity to comparator; state greater/less/equal, units, denominators and period in qualification.', 'contrast'),
    'trade_off': ('comparison', 'A benefit or objective to its associated cost or competing objective; do not invent a numerical exchange rate.', 'contrast'),
    'alternative_explanation': ('comparison', 'One explanation to a competing account of the same phenomenon; do not assert either is established.', 'alternative'),
    'substitution': ('comparison', 'Substitute to what it can replace; retain when and how far replacement is possible.', 'alternative'),
    'complementarity': ('comparison', 'One input or arrangement to another whose usefulness it enhances; not mere co-occurrence.', 'association'),
    'evidence_for': ('evidential', 'Evidence to the proposition it supports; support is not proof.', 'evidence_for'),
    'evidence_against': ('evidential', 'Evidence to the proposition it weakens or challenges; distinguish weakened from refuted.', 'evidence_for'),
    'tests': ('evidential', 'Test design or reported test to the hypothesis examined; record the outcome, if given, in qualification.', 'evidence_for'),
    'robustness_check': ('evidential', 'Alternative-specification or sample result to the focal finding it checks; do not assume the check succeeds.', 'evidence_for'),
    'example_of': ('expository', 'Specific case to the general idea it illustrates; an example does not establish prevalence.', 'evidence_for'),
    'defines': ('expository', 'Definition to the concept-bearing point it clarifies; preserve the defining boundaries.', 'definition_part'),
    'operationalizes': ('expository', 'Measurement or proxy rule to the theoretical concept it measures; proxy need not equal concept.', 'definition_part'),
    'part_of': ('expository', 'Component or subcategory to the whole; preserve exhaustive versus partial coverage.', 'definition_part'),
    'elaborates': ('expository', 'Added detail to the more general point; not new independent evidence by itself.', 'definition_part'),
    'scope_restriction': ('qualification', 'Restriction to the proposition whose applicability it limits.', 'qualification'),
    'exception': ('qualification', 'Exceptional case to the general pattern it falls outside; preserve the size/scope of the exception.', 'qualification'),
    'concession': ('qualification', 'Acknowledged consideration to a maintained conclusion; not automatic refutation or a reversal.', 'qualification'),
    'temporal_sequence': ('temporal', 'Earlier event to later event; sequence alone does not imply causation.', 'temporal_sequence'),
    'change_over_time': ('temporal', 'Earlier state to later state; preserve the change, period and comparison basis.', 'temporal_sequence'),
    'reason_for_argument': ('argumentative', 'Stated reason or premise to the assessment or recommendation it motivates; distinguish a reason from empirical evidence.', 'evidence_for'),
    'goal_means': ('argumentative', 'Proposed action or instrument to the stated objective; does not assert success or verified author intent.', 'condition'),
}
RELATIONS_V3 = tuple(RELATION_SPEC)


def validate_brief_v3(row):
    if not isinstance(row, dict):
        raise ValueError('brief_record')
    checked = copy.deepcopy(row)
    relations = checked.get('brief', {}).get('relationships') if isinstance(checked.get('brief'), dict) else None
    if not isinstance(relations, list):
        raise ValueError('brief_relations')
    for relation in relations:
        if not isinstance(relation, dict) or not isinstance(relation.get('type'), str) or relation['type'] not in RELATION_SPEC:
            raise ValueError('v3_relation_type')
        relation['type'] = RELATION_SPEC[relation['type']][2]
    validate_brief_v2(checked)
    return row


def reconstruction_conditions(row):
    """Matched pilot payloads: keep identical content; vary only explicit guidance."""
    validate_brief_v3(row)
    common = {key: copy.deepcopy(row['brief'][key]) for key in ('task', 'key_points', 'anchors')}
    core = dict(id=row['id'], target_sha256=row['target_sha256'], brief=common,
                brief_status=row['status'], brief_notes=copy.deepcopy(row['notes']))
    guided = copy.deepcopy(core)
    guided['brief'] = copy.deepcopy(row['brief'])
    return {'content_only': core, 'purpose_and_relations': guided}
