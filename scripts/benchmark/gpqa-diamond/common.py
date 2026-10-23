"""Shared GPQA-Diamond benchmark pieces (questions, prompt, extractor).

Used by compute_base_llm.py and compute_rjepa.py so both rows share the
identical question set, option shuffling, prompt text and answer
extraction.  The directory name has a hyphen, so siblings import this
via ``sys.path`` + ``import common`` rather than a package import.
"""
import re

BENCHMARK = "gpqa-diamond"

# Shared answer-format instruction: identical across ALL comparison rows
# (base LLM, rjepa, future cot/none baselines) - the standard practice
# (cf. the MATH benchmark's \boxed{} prompt).  Carries no answer
# information, only the output format.
FORMAT_INSTRUCTION = (
    "End your response with your final answer in the format \\boxed{X}, "
    "where X is the letter of the chosen option."
)


def load_questions(args) -> tuple:
    """Return ([{'question', 'options', 'gold'}...], source).

    Sources, in order:
      1. ``--dataset-path`` (local HF datasets dir)
      2. HF ``Idavidrein/gpqa`` (gated - works when HF_TOKEN is set)
    """
    from datasets import load_dataset

    if args.dataset_path:
        rows_raw = load_dataset(args.dataset_path, split="train")
        source = "local_dataset_path"
    else:
        rows_raw = load_dataset("Idavidrein/gpqa", "gpqa_diamond",
                                split="train")
        source = "hf_gated"
        print("loaded questions from HF Idavidrein/gpqa", flush=True)

    rows = []
    for ex in rows_raw:
        # HF dataset uses capitalized field names ("Question",
        # "Correct Answer", "Incorrect Answer 1/2/3"); any other source
        # may use lowercase / a list.  Normalize both here.
        q = ex.get("question") or ex.get("Question")
        correct = ex.get("correct_answer") or ex.get("Correct Answer")
        if "incorrect_answers" in ex:
            incorrect = list(ex["incorrect_answers"])
        else:
            incorrect = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"],
                         ex["Incorrect Answer 3"]]
        opts = [correct] + list(incorrect)
        # deterministic shuffle so gold-letter mapping is reproducible
        import random
        rng = random.Random(args.seed + len(rows))
        rng.shuffle(opts)
        letters = "ABCD"
        options = list(zip(letters, opts))
        gold = letters[opts.index(correct)]
        rows.append({"question": q, "options": options, "gold": gold})
    if args.limit:
        rows = rows[: args.limit]
    return rows, source


def build_prompt(q_row) -> str:
    lines = [q_row["question"], "", "Options:"]
    for letter, text in q_row["options"]:
        lines.append(f"{letter}) {text}")
    lines += ["", FORMAT_INSTRUCTION]
    return "\n".join(lines)


ANSWER_PATTERNS = [
    # LaTeX boxed answer - the standard math final-answer format
    re.compile(r"\\boxed\{([A-D])\}"),
    re.compile(r"answer\s*(?:is|:)?\s*[:：]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"correct\s+(?:answer|option|choice)\s*(?:is|:)?\s*[:：]?\s*\(?([A-D])\)?",
               re.IGNORECASE),
    re.compile(r"\b([A-D])\b\s*\.?\s*$"),
]


def extract_answer(text: str, options=None):
    """Extract the final answer letter, preferring the post-think section.

    Priority:
      1. boxed LaTeX (\\boxed{X})
      2. "answer/correct ... X" phrases
      3. option-TEXT match: the model often answers by writing out the
         option's content verbatim instead of a letter (options: dict
         letter -> text).  The matched option whose text occurs closest
         to the end wins.
      4. last-resort: last standalone A-D letter in the section.
    """
    post = text.rsplit("</think>", 1)[-1] if "</think>" in text else text

    def _norm(t):
        return re.sub(r"\s+", "", str(t)).lower()

    for section in (post, text):
        for pat in ANSWER_PATTERNS:
            matches = pat.findall(section)
            if matches:
                return matches[-1].upper()
        if options:
            sec_norm = _norm(section)
            hits = []
            for letter, opt_text in options.items():
                t = _norm(opt_text)
                if len(t) >= 8 and t in sec_norm:
                    hits.append((sec_norm.rfind(t), letter))
            if hits:
                # the option mentioned last is the conclusion
                return max(hits)[1].upper()
        # last resort: last standalone A-D letter in this section
        loose = re.findall(r"\b([A-D])\b", section)
        if loose:
            return loose[-1]
    return None
