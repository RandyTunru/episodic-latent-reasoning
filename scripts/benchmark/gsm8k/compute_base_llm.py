"""Base-LLM baseline for GSM8K (Block A / zero-shot column).

Evaluates the LLM (default Qwen/Qwen3-4B, or an adapted/fine-tuned
checkpoint via ``--checkpoint``) on the 1,319 GSM8K test questions with
native thinking enabled.  Metrics: exact numeric match on the ``#### N``
convention (the standard GSM8K metric) plus per-question latency and
decode-token counts for the efficiency table.

Output: ``logs/benchmark/gsm8k/<run>/`` with meta.json, predictions.jsonl
(written live, one line per completed batch) and report.md.

Run:
    python scripts/benchmark/gsm8k/compute_base_llm.py \
        [--checkpoint artifacts/.../adapted_llm.pt] [--limit 10]
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BENCHMARK, load_questions, build_prompt, extract_answer, answers_match,
)
from tools.benchmark_report import (
    new_run_dir, write_report, timing_summary, format_timing_md,
)
from tools.telegram_service import send_bot_message
from tools.on_error import format_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B",
                   help="base LLM (the adaptation column fine-tunes from this)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="optional adapted/fine-tuned LLM checkpoint (.pt with "
                        "'model_state_dict' or a bare state dict); loaded on "
                        "top of --model-id")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="optional local HF datasets dir (defaults to "
                        "openai/gsm8k)")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="generation budget (thinking + answer)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="questions per generate call (greedy decode is "
                        "batch-invariant); 1 for honest per-question latency")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: evaluate only the first N questions")
    p.add_argument("--run-name", type=str, default=None,
                   help="output folder name (default: base_llm_<timestamp>)")
    p.add_argument("--no-thinking", action="store_true",
                   help="disable the think block (ablation)")
    return p.parse_args()


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = new_run_dir(BENCHMARK, "base_llm", args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    ).to(device)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu",
                        weights_only=False)
        sd = ck.get("model_state_dict", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.checkpoint} (missing {len(missing)}, "
              f"unexpected {len(unexpected)})", flush=True)
    model.eval()

    questions, source = load_questions(args)
    print(f"{len(questions)} questions (source: {source}), "
          f"output -> {out_dir}", flush=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only models must left-pad: right-padding puts pad tokens
    # after the content, which the model attends to as context
    tokenizer.padding_side = "left"

    predictions = []
    preds_file = open(out_dir / "predictions.jsonl", "a")  # grows live
    t0 = time.time()
    for start in range(0, len(questions), args.batch_size):
        chunk = questions[start:start + args.batch_size]
        convs = [[{"role": "user", "content": build_prompt(q)}]
                 for q in chunk]
        enc = tokenizer.apply_chat_template(
            convs, add_generation_prompt=True,
            enable_thinking=not args.no_thinking,
            return_tensors="pt", padding=True,
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

        # generation appends AFTER the padded input length for every row
        L_max = input_ids.size(1)
        for j, q in enumerate(chunk):
            i = start + j
            cont = out[j, L_max:]
            eos_pos = (cont == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            n_new = int(eos_pos[0]) + 1 if eos_pos.numel() else cont.size(0)
            raw = tokenizer.decode(cont[:n_new], skip_special_tokens=False)
            # an answer can only be extracted from a COMPLETED generation
            incomplete = None
            if eos_pos.numel() == 0:
                incomplete = "budget_cut"
            elif "<think>" in raw and "</think>" not in raw:
                incomplete = "unclosed_think"
            if incomplete:
                pred = None
            else:
                pred = extract_answer(raw)
            rec = {
                "idx": i,
                "question": q["question"],
                "gold": q["gold"],
                "predicted": pred,
                "correct": answers_match(pred, q["gold"]),
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
        "checkpoint": args.checkpoint,
        "n_questions": n,
        "questions_source": source,
        "accuracy": round(n_correct / n, 4) if n else None,
        "n_correct": n_correct,
        "n_parse_failures": n_parse_fail,
        "n_incomplete": n_incomplete,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "format_instruction": "gsm8k_hashes",
        "latency_note": ("per-question latency = amortized batch wall time"
                         if args.batch_size > 1
                         else "per-question wall time (batch-1)"),
        "timing": ts,
        "elapsed_s": round(elapsed, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report = (
        f"# GSM8K — base LLM baseline\n\n"
        f"- model: `{args.model_id}`"
        + (f" + checkpoint `{args.checkpoint}`" if args.checkpoint else "")
        + f"\n- thinking: {'off' if args.no_thinking else 'on'}\n"
        f"- questions: {n} (source: `{source}`), "
        f"generated: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"## Accuracy\n\n"
        f"**{n_correct}/{n} = {n_correct / n:.1%}** exact numeric match "
        f"(#### N); parse failures: {n_parse_fail}; incomplete generations "
        f"(counted wrong): {n_incomplete}\n\n"
        f"## Timing\n\n"
        + format_timing_md(ts)
        + (f"\n(batch_size={args.batch_size}: latency is amortized batch "
           f"wall time; for honest per-question latency rerun with "
           f"--batch-size 1)\n" if args.batch_size > 1 else "")
    )

    write_report(out_dir, meta, None, report)  # predictions.jsonl was live
    print(f"\nacc={n_correct / n:.3f} | wrote {out_dir}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
        send_bot_message(
            f"GSM8K base-LLM baseline finished:\n"
            f"Model: {args.model_id}\n"
            f"Output: logs/benchmark/{BENCHMARK}/"
        )
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"GSM8K base-LLM baseline failed:\n{err_msg}")
        raise
