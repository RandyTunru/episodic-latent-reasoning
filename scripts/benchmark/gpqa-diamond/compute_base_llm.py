"""Base-LLM baseline for GPQA-Diamond (Block A / zero-shot column).

Evaluates the untrained base LLM (default Qwen/Qwen3-4B - the SAME model
the latent-projection row uses) on the 198 GPQA-Diamond multiple-choice
questions, with native thinking enabled.  This is the zero-shot CoT
baseline: no training of ours, no corpus SFT.

Metrics: exact-match accuracy on the final answer choice (the standard
GPQA metric), plus per-question latency and decode-token counts for the
efficiency table.  Questions come from HF ``Idavidrein/gpqa``
(``gpqa_diamond`` split); answer options are shuffled with a fixed seed
so the gold-letter mapping is reproducible and recorded.

Output: ``logs/benchmark/gpqa-diamond/<run>/`` with meta.json,
predictions.jsonl and report.md (via tools.benchmark_report).  The
published leaderboard CSV under data/benchmark/gpqa-diamond/test/ is
read for context in the report.

Run:
    python scripts/benchmark/gpqa-diamond/compute_base_llm.py \
        --model-id Qwen/Qwen3-4B [--limit 10] [--run-name my_run]
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # hyphenated dir

from common import (
    BENCHMARK, load_questions, build_prompt, extract_answer,
)
from tools.benchmark_report import (
    new_run_dir, write_report, timing_summary, format_timing_md,
)
from tools.telegram_service import send_bot_message
from tools.on_error import format_error

BENCHMARK = "gpqa-diamond"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B",
                   help="LLM to evaluate (must match the model the "
                        "latent-projection row was trained on)")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="optional local dir holding the gpqa_diamond split "
                        "(defaults to HF Idavidrein/gpqa)")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="generation budget (thinking + answer)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="questions per generate call.  Greedy decoding is "
                        "batch-invariant, so accuracy is unchanged; batches "
                        "amortize the per-step loop overhead (~20-30x wall-"
                        "clock speedup).  Latency recorded per question is "
                        "the batch wall time (amortized).  Use 1 for honest "
                        "per-question latency.")
    p.add_argument("--seed", type=int, default=0,
                   help="option-shuffle seed (fixes gold-letter mapping)")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: evaluate only the first N questions")
    p.add_argument("--run-name", type=str, default=None,
                   help="output folder name (default: base_llm_<timestamp>)")
    p.add_argument("--no-thinking", action="store_true",
                   help="disable the <think>...</think> section in the "
                        "prompt (for ablation)")
    return p.parse_args()


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = new_run_dir(BENCHMARK, "base_llm", args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    questions, source = load_questions(args)
    print(f"{len(questions)} questions (source: {source}), "
          f"output -> {out_dir}", flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    predictions = []
    preds_file = open(out_dir / "predictions.jsonl", "a")  # grows live
    t0 = time.time()
    for start in range(0, len(questions), args.batch_size):
        chunk = questions[start:start + args.batch_size]

        convs = [[{"role": "user", "content": build_prompt(q)}]
                 for q in chunk]
        
        enc = tokenizer.apply_chat_template(
            convs, add_generation_prompt=True, enable_thinking=not args.no_thinking,
            return_tensors="pt", padding=True, padding_side="left"
        )
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        t_start = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        batch_time = time.time() - t_start

        L_max = input_ids.size(1)
        for j, q in enumerate(chunk):
            i = start + j
            cont = out[j, L_max:]
            eos_pos = (cont == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            n_new = int(eos_pos[0]) + 1 if eos_pos.numel() else cont.size(0)
            raw = tokenizer.decode(cont[:n_new], skip_special_tokens=False)
            # An answer can only be extracted from a COMPLETED generation:
            # no EOS = hit the token budget mid-generation; a <think> that
            # never closes = the model never finished thinking.  Both are
            # incomplete - mark them instead of guessing letters out of a
            # degenerate loop (e.g. idx 197: 16k tokens of loop, no </think>).
            incomplete = None
            if eos_pos.numel() == 0:
                incomplete = "budget_cut"
            elif "<think>" in raw and "</think>" not in raw:
                incomplete = "unclosed_think"
            if incomplete:
                pred = None
            else:
                pred = extract_answer(
                    raw, options={l: t for l, t in q["options"]})
            rec = {
                "idx": i,
                "question": q["question"],
                "options": {l: t for l, t in q["options"]},
                "gold": q["gold"],
                "predicted": pred,
                "correct": pred == q["gold"],
                "incomplete": incomplete,
                "latency_s": round(batch_time, 3),  # amortized batch wall time
                "decode_tokens": n_new,
                "raw": raw,
            }
            predictions.append(rec)
            preds_file.write(json.dumps(rec) + "\n")
            preds_file.flush()
        n_done = len(predictions)
        print(f"[{n_done}/{len(questions)}] acc so far "
              f"{sum(p['correct'] for p in predictions) / n_done:.3f}",
              flush=True)

    preds_file.close()
    elapsed = time.time() - t0
    n = len(predictions)
    n_correct = sum(p["correct"] for p in predictions)
    n_parse_fail = sum(p["predicted"] is None and p.get("incomplete") is None
                       for p in predictions)
    n_incomplete = sum(p.get("incomplete") is not None for p in predictions)
    ts = timing_summary([p["latency_s"] for p in predictions],
                        [p["decode_tokens"] for p in predictions])

    meta = {
        "benchmark": BENCHMARK,
        "row": "base_llm_thinking" if not args.no_thinking else "base_llm_no_thinking",
        "model_id": args.model_id,
        "n_questions": n,
        "questions_source": source,
        "accuracy": round(n_correct / n, 4) if n else None,
        "n_correct": n_correct,
        "n_parse_failures": n_parse_fail,
        "n_incomplete": n_incomplete,
        "max_new_tokens": args.max_new_tokens,
        "format_instruction": "boxed",
        "batch_size": args.batch_size,
        "latency_note": ("per-question latency = amortized batch wall time"
                         if args.batch_size > 1
                         else "per-question wall time (batch-1)"),
        "seed": args.seed,
        "timing": ts,
        "elapsed_s": round(elapsed, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    source_note = ""

    # published leaderboard context (optional, from the local CSV)
    leaderboard_note = ""
    lb_csv = (REPO_ROOT / "data" / "benchmark" / BENCHMARK / "test" /
              "gpqa_diamond.csv")
    if lb_csv.exists():
        import csv
        with open(lb_csv) as f:
            rows = list(csv.DictReader(f))
        scores = [float(r["mean_score"]) for r in rows if r.get("mean_score")]
        if scores:
            best = max(rows, key=lambda r: float(r.get("mean_score") or 0))
            leaderboard_note = (
                f"\n## Published leaderboard context\n"
                f"Local leaderboard CSV: {len(rows)} entries; "
                f"best published mean score **{max(scores):.3f}** "
                f"({best.get('Model version')}); median {sorted(scores)[len(scores) // 2]:.3f}.\n"
            )

    report = (
        f"# GPQA-Diamond — base LLM (thinking) baseline\n\n"
        f"- model: `{args.model_id}` (untrained by us, native thinking on)\n"
        f"- questions: {n} (source: `{source}`), generated: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"{source_note}\n"
        f"## Accuracy\n\n"
        f"**{n_correct}/{n} = {n_correct / n:.1%}** exact match "
        f"(final answer choice); parse failures: {n_parse_fail}; "
        f"incomplete generations (budget-cut or unclosed think, counted "
        f"as wrong): {n_incomplete}\n\n"
        f"## Timing\n\n"
        + format_timing_md(ts)
        + (f"\n(batch_size={args.batch_size}: latency is amortized batch "
           f"wall time per question; for honest per-question latency rerun "
           f"with --batch-size 1)\n"
           if args.batch_size > 1 else "")
        + leaderboard_note
    )

    write_report(out_dir, meta, None, report)  # predictions.jsonl was written live
    print(f"\nacc={n_correct / n:.3f} | mean latency {ts['latency_mean_s']}s "
          f"| wrote {out_dir}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
        send_bot_message(
            f"GPQA-Diamond base-LLM baseline finished:\n"
            f"Model: {args.model_id}\n"
            f"Output: logs/benchmark/{BENCHMARK}/"
        )
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"GPQA-Diamond base-LLM baseline failed:\n{err_msg}")
        raise
