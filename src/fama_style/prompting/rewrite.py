"""Frozen prompt formatting used by historical experiments."""

REWRITE_INSTRUCTION = (
    "Rewrite the passage below as formal academic economics prose. Preserve every "
    "substantive claim and qualification. Do not add facts, examples, or commentary. "
    "Return only the rewritten passage, without Markdown."
)

def generation_prompt(plain_english, instruction=False):
    if not instruction:
        return plain_english
    return f"{REWRITE_INSTRUCTION}\n\nPassage:\n{plain_english}"
