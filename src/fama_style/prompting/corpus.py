"""Frozen multi-paper v1 instruction, identical to the curated data export."""

FORMAT = "fama_multi_paper_v1"
INSTRUCTION = (
    "Rewrite the passage below in the formal academic economics style of Eugene Fama "
    "and his coauthors. Preserve every substantive claim, actor, attribution, "
    "condition, qualification, uncertainty, negation, quantity, comparison, and "
    "causal relationship. Do not add or omit claims or strengthen tentative claims. "
    "Return only one coherent paragraph, without headings, Markdown, explanations, "
    "or commentary."
)


def make_prompt(text):
    return f"{INSTRUCTION}\n\nPassage:\n{text}"
