"""Frozen prompt formatting used by historical experiments."""

INSTRUCTION = (
    "Rewrite the passage in the academic economics style of Eugene Fama and his "
    "coauthored Fama-Jensen writing, as demonstrated by the examples below. "
    "Use precise economic terminology, direct analytical statements, explicit causal "
    "or conditional relationships where present, and concise definitions. Preserve "
    "every substantive claim, qualification, uncertainty, negation, and attribution. "
    "Preserve who acts and who bears risk. Use specialized terms only when they "
    "express the same meaning. Do not import facts, names, examples, citations, or "
    "claims from the demonstrations. Do not add claims or strengthen tentative "
    "claims. Return only one coherent rewritten paragraph, without headings, "
    "Markdown, explanation, or commentary."
)

def make_prompt(demos, source):
    parts = [INSTRUCTION]
    for i, demo in enumerate(demos, 1):
        parts.append(f"Example {i}\nInput:\n{demo['source']}\nRewrite:\n{demo['target']}")
    parts.append(f"Now rewrite this passage:\n{source}\n\nRewrite:")
    return "\n\n".join(parts)
