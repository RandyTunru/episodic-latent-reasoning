"""Shared GSM8K benchmark pieces (questions, prompt, extractor).

Used by compute_base_llm.py and compute_rjepa.py so both rows share the
identical question set, prompt text and answer extraction.  The
canonical GSM8K convention: solutions end with ``#### N``; the number
after the hashes is the graded answer.

Golds are extracted from the dataset's ``answer`` column (the official
step-by-step solution text, which always ends ``#### N``), so the
evaluation needs no external label source.
"""
import re

BENCHMARK = "gsm8k"

# Shared answer-format instruction: identical across ALL comparison rows.
# The standard GSM8K output convention - carries no answer information,
# only the output format.
FORMAT_INSTRUCTION = (
    "Solve the problem step by step and end your solution with "
    "'#### X', and X should be the final numeric answer. "
    "For example, if the answer is 42, you should end your solution with '#### 42'."
)


def _norm_number(s: str):
    """Normalize a numeric string for comparison: strip commas/spaces
    and a trailing decimal point."""
    s = str(s).strip().replace(",", "").replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_hashes_number(text: str):
    """Extract the number after ``####`` in *text*, normalized."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    return _norm_number(m.group(1)) if m else None


def answers_match(pred, gold) -> bool:
    """Numeric equality with float tolerance; None never matches."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def load_questions(args) -> tuple:
    """Return ([{'question', 'gold'}...], source).  Gold = the numeric
    answer extracted from the official solution's ``#### N``."""
    from datasets import load_dataset

    path = args.dataset_path or "openai/gsm8k"
    ds = load_dataset(path, "main", split="test")
    rows = []
    for ex in ds:
        gold = extract_hashes_number(ex["answer"])
        rows.append({"question": ex["question"], "gold": gold})
    if args.limit:
        rows = rows[: args.limit]
    return rows, "hf_" + path.replace("/", "_")


def build_prompt(q_row) -> str:
    return q_row["question"] + "\n\n" + FORMAT_INSTRUCTION


def extract_answer(text: str):
    """Extract the final numeric answer (``#### N``), preferring the
    post-think section."""
    post = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    for section in (post, text):
        m = extract_hashes_number(section)
        if m:
            return m
    return None
