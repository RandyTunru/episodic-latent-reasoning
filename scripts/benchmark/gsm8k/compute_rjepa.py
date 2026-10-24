"""RJEPA system row for GSM8K (zero-shot column; adapted column by
pointing the checkpoint flags at adapted checkpoints).

The full pipeline on the same 1,319 GSM8K test questions as
compute_base_llm.py:

  1. ctx encoding   - the question prompt is encoded by the frozen
                      Qwen3-4B encoder, mirroring the precompute's ctx
                      branch (per-token last_hidden_state, bf16).
  2. latent chain   - the predictor generates the latent reasoning
                      chain free-running until the router halts or the
                      cap is reached.
  3. LLM answer     - the frozen Qwen3-4B + projection consumes
                      [instruction | <think> proj(latents) </think>] and
                      generates the answer (no thinking: the latents ARE
                      the reasoning).

Zero-shot vs adapted is ONLY a matter of which checkpoints the flags
point at: zero-shot uses the MoT predictor + MoT-latent projection;
adapted uses the GSM8K-train-fine-tuned predictor + the projection
retrained on its latents.  ctx/latent stages are cached under
data/benchmark/gsm8k/cache/ keyed by their inputs.

Output: ``logs/benchmark/gsm8k/<run>/`` with meta.json, predictions.jsonl
(written live) and report.md.

Run:
    python scripts/benchmark/gsm8k/compute_rjepa.py [--limit 10]
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BENCHMARK, load_questions, build_prompt, extract_answer, answers_match,
)
from scripts.openr1.data.openr1_latent_precompute import generate_batch
from src.models.r_jepa.r_jepa import CausalRJEPA
from tools.benchmark_report import (
    new_run_dir, write_report, timing_summary, format_timing_md,
)
from tools.telegram_service import send_bot_message
from tools.on_error import format_error

PREDICTOR_CONFIG = "config/openr1_all/qwen3_4b_rewritten_rjepa.yaml"
CACHE_DIR = REPO_ROOT / "data" / "benchmark" / "gsm8k" / "cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B",
                   help="frozen LLM (must match the projection)")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="optional local HF datasets dir (defaults to "
                        "openai/gsm8k)")
    p.add_argument("--predictor-checkpoint", type=str,
                   default=("artifacts/checkpoints/predictor/"
                            "qwen3_llm_4b_rjepa_openr1_all_rewritten_03/"
                            "model_step_55000.pt"),
                   help="predictor checkpoint (MoT for zero-shot; adapted "
                        "for the adaptation column)")
    p.add_argument("--projection-checkpoint", type=str,
                   default=("artifacts/checkpoints/projector/"
                            "llm_latent_pilot_01/projection_step_005000.pt"),
                   help="projection checkpoint (MoT-latent for zero-shot; "
                        "GSM8K-latent for the adaptation column)")
    # p.add_argument("--predictor-config", type=str, default=PREDICTOR_CONFIG,
    #                help="predictor architecture config yaml")
    p.add_argument("--max-steps", type=int, default=64,
                   help="predictor generation cap")
    p.add_argument("--halt-threshold", type=float, default=0.0,
                   help="router halt rule: first step with logit > tau")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="LLM answer budget (no thinking: latents are the "
                        "reasoning)")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: evaluate only the first N questions")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore cached ctx/latent stages and recompute")
    p.add_argument("--run-name", type=str, default=None,
                   help="output folder name (default: rjepa_<timestamp>)")
    return p.parse_args()


def ctx_cache_path() -> Path:
    return CACHE_DIR / "ctx_qwen3-4b.parquet"


def latent_cache_path(ckpt: str) -> Path:
    return CACHE_DIR / f"latents_{Path(ckpt).stem}.parquet"


CTX_SCHEMA = pa.schema([
    ("idx", pa.int32()),
    ("ctx_len", pa.int32()),
    ("ctx_embeddings", pa.binary()),
])

LATENT_SCHEMA = pa.schema([
    ("idx", pa.int32()),
    ("num_steps", pa.int32()),
    ("halted", pa.int8()),
    ("step_latents", pa.binary()),
])


def encode_ctxs(config, questions, device):
    """Frozen Qwen3-4B token-level encodings of the question prompts
    (mirrors the precompute ctx branch)."""
    max_ctx = config["encoder_max_seq_len"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model.eval()

    rows = []
    B = 8
    with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, len(questions), B):
            batch = questions[start:start + B]
            enc = tokenizer(
                [build_prompt(q) for q in batch], truncation=True,
                max_length=max_ctx, padding=True, return_tensors="pt",
            )
            hidden = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).last_hidden_state.cpu()
            for i in range(len(batch)):
                n = int(enc["attention_mask"][i].sum())
                blob = hidden[i, :n].view(torch.uint8).numpy().tobytes()
                rows.append({"idx": start + i, "ctx_len": n,
                             "ctx_embeddings": blob})
    return rows


def build_predictor(ck, device):
    config = ck["config"]
    kwargs = dict(
        encoder_dim=config["encoder_hidden_size"],
        predictor_dim=config["predictor_hidden_size"],
        num_heads=config["predictor_num_heads"],
        d_ff=config["predictor_d_ff"],
        num_layers=config["predictor_num_layers"],
        max_seq_length=config["predictor_max_seq_len"],
        dropout=0.0,
    )
    model = CausalRJEPA(
        predictor_kwargs=kwargs, encoder=None,
        encoder_max_seq_length=config["encoder_max_seq_len"],
        encoder_pooling=config["encoder_pooling"],
    ).to(device)

    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model.predictor


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = new_run_dir(BENCHMARK, "rjepa", args.run_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.predictor_checkpoint, map_location="cpu",
                    weights_only=False)
    config = ck["config"]

    questions, source = load_questions(args)
    print(f"{len(questions)} questions (source: {source}), "
          f"output -> {out_dir}", flush=True)

    # ---- stage 1: ctx encoding (cached) ----
    ctx_path = ctx_cache_path()
    if ctx_path.exists() and not args.no_cache:
        ctx_rows = pq.read_table(ctx_path).to_pylist()
        print(f"ctx cache hit: {len(ctx_rows)} rows", flush=True)
    else:
        print("encoding ctx with frozen Qwen3-4B ...", flush=True)
        t0 = time.time()
        ctx_rows = encode_ctxs(config, questions, device)
        pq.write_table(pa.Table.from_pylist(ctx_rows, schema=CTX_SCHEMA),
                       ctx_path, row_group_size=64)
        print(f"ctx encoded: {len(ctx_rows)} rows in {time.time() - t0:.0f}s",
              flush=True)

    # ---- stage 2: latent generation (cached by checkpoint) ----
    lat_path = latent_cache_path(args.predictor_checkpoint)
    if lat_path.exists() and not args.no_cache:
        lat_rows = pq.read_table(lat_path).to_pylist()
        print(f"latent cache hit: {len(lat_rows)} rows", flush=True)
    else:
        predictor = build_predictor(ck, device)
        print("generating latent chains (free-running predictor) ...",
              flush=True)
        t0 = time.time()
        lat_rows = []
        B = 8
        for start in range(0, len(questions), B):
            ctxs = [torch.frombuffer(r["ctx_embeddings"],
                                     dtype=torch.bfloat16).reshape(
                -1, config["encoder_hidden_size"])
                for r in ctx_rows[start:start + B]]
            chains = generate_batch(predictor, ctxs, device,
                                    args.halt_threshold, args.max_steps)
            for i, (blob_lat, _blob_log, halted) in enumerate(chains):
                lat_rows.append({
                    "idx": start + i,
                    "num_steps": len(blob_lat) // (config["encoder_hidden_size"] * 2),
                    "halted": halted,
                    "step_latents": blob_lat,
                })
        pq.write_table(pa.Table.from_pylist(lat_rows, schema=LATENT_SCHEMA),
                       lat_path, row_group_size=64)
        n_halt = sum(r["halted"] for r in lat_rows)
        print(f"latents generated: {len(lat_rows)} rows "
              f"({n_halt} halted) in {time.time() - t0:.0f}s", flush=True)

    # ---- stage 3: LLM + projection answer generation ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    embed_fn = model.get_input_embeddings()
    H = model.config.hidden_size

    E = config["encoder_hidden_size"]
    proj = nn.Linear(E, H, dtype=torch.bfloat16).to(device)
    proj_ck = torch.load(args.projection_checkpoint, map_location="cpu",
                         weights_only=False)["projection"]
    proj.load_state_dict({k: v.to(torch.bfloat16) for k, v in proj_ck.items()})

    think_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")

    predictions = []
    preds_file = open(out_dir / "predictions.jsonl", "a")  # grows live
    t0 = time.time()
    for i, q in enumerate(questions):
        t_q0 = time.time()
        # chat template with the empty think block (Qwen3 non-thinking
        # mechanism), split at the tags - the SAME structure the
        # projection was trained with
        tpl = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(q)}],
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        o = [k for k, t in enumerate(tpl) if t == think_id]
        c = [k for k, t in enumerate(tpl) if t == close_id]
        pre_ids = tpl[:o[0] + 1]
        post_ids = tpl[c[0]:]

        lat = torch.frombuffer(
            lat_rows[i]["step_latents"], dtype=torch.bfloat16,
        ).reshape(-1, E).to(device)
        t_lat0 = time.time()
        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            proj_lat = proj(lat)
        lat_time = time.time() - t_lat0

        n_pre, n_lat, n_post = len(pre_ids), proj_lat.size(0), len(post_ids)
        total_in = n_pre + n_lat + n_post
        embeds = torch.zeros(1, total_in, H, dtype=torch.bfloat16,
                             device=device)
        with torch.no_grad():
            embeds[0, :n_pre] = embed_fn(torch.tensor(pre_ids, device=device))
            embeds[0, n_pre + n_lat:] = embed_fn(torch.tensor(
                post_ids, device=device))
        embeds[0, n_pre:n_pre + n_lat] = proj_lat
        mask = torch.ones(1, total_in, dtype=torch.long, device=device)

        t_llm0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embeds, attention_mask=mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        llm_time = time.time() - t_llm0
        total_time = time.time() - t_q0

        # PROVEN (HF 5.14): with inputs_embeds, generate returns ONLY the
        # newly generated tokens - slice at 0, count to the row's EOS.
        cont = out[0]
        eos_pos = (cont == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        n_new = int(eos_pos[0]) + 1 if eos_pos.numel() else cont.size(0)
        raw = tokenizer.decode(cont[:n_new], skip_special_tokens=False)
        incomplete = "budget_cut" if eos_pos.numel() == 0 else None
        pred = (extract_answer(raw)
                if incomplete is None else None)
        rec = {
            "idx": i,
            "question": q["question"],
            "gold": q["gold"],
            "predicted": pred,
            "correct": answers_match(pred, q["gold"]),
            "incomplete": incomplete,
            "num_steps": lat_rows[i]["num_steps"],
            "halted": bool(lat_rows[i]["halted"]),
            "latent_gen_s": round(lat_time, 3),
            "llm_gen_s": round(llm_time, 3),
            "total_s": round(total_time, 3),
            "decode_tokens": n_new,
            "raw": raw,
        }
        predictions.append(rec)
        preds_file.write(json.dumps(rec) + "\n")
        preds_file.flush()
        if (i + 1) % 50 == 0 or i + 1 == len(questions):
            print(f"[{i + 1}/{len(questions)}] acc so far "
                  f"{sum(p['correct'] for p in predictions) / (i + 1):.3f}",
                  flush=True)

    preds_file.close()
    elapsed = time.time() - t0
    n = len(predictions)
    n_correct = sum(p["correct"] for p in predictions)
    n_parse_fail = sum(p["predicted"] is None and p.get("incomplete") is None
                       for p in predictions)
    n_incomplete = sum(p.get("incomplete") is not None for p in predictions)
    n_halt = sum(p["halted"] for p in predictions)
    ts = timing_summary([p["total_s"] for p in predictions],
                        [p["decode_tokens"] for p in predictions])

    meta = {
        "benchmark": BENCHMARK,
        "row": "rjepa_zero_shot",
        "model_id": args.model_id,
        "predictor_checkpoint": args.predictor_checkpoint,
        "projection_checkpoint": args.projection_checkpoint,
        "questions_source": source,
        "halt_threshold": args.halt_threshold,
        "max_steps": args.max_steps,
        "n_questions": n,
        "accuracy": round(n_correct / n, 4) if n else None,
        "n_correct": n_correct,
        "n_parse_failures": n_parse_fail,
        "n_incomplete": n_incomplete,
        "n_halted": n_halt,
        "n_capped": n - n_halt,
        "chain_len_mean": round(float(np.mean(
            [p["num_steps"] for p in predictions])), 1),
        "max_new_tokens": args.max_new_tokens,
        "format_instruction": "gsm8k_hashes",
        "timing": ts,
        "elapsed_s": round(elapsed, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report = (
        f"# GSM8K — RJEPA system row (zero-shot)\n\n"
        f"- LLM: `{args.model_id}` frozen + projection "
        f"`{Path(args.projection_checkpoint).name}`\n"
        f"- predictor: `{Path(args.predictor_checkpoint).name}`\n"
        f"- questions: {n} (source: `{source}`), "
        f"generated: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- halt: tau={args.halt_threshold}, cap={args.max_steps}; "
        f"halted chains: {n_halt}/{n}\n\n"
        f"## Accuracy\n\n"
        f"**{n_correct}/{n} = {n_correct / n:.1%}** exact numeric match "
        f"(#### N); parse failures: {n_parse_fail}; incomplete generations "
        f"(counted wrong): {n_incomplete}\n\n"
        f"## Timing\n\n"
        + format_timing_md(ts)
        + f"\nLatency breakdown per question (mean): latent-gen "
        f"{np.mean([p['latent_gen_s'] for p in predictions]):.3f}s, "
        f"LLM decode {np.mean([p['llm_gen_s'] for p in predictions]):.3f}s, "
        f"total {np.mean([p['total_s'] for p in predictions]):.3f}s.\n"
    )

    write_report(out_dir, meta, None, report)  # predictions.jsonl was live
    print(f"\nacc={n_correct / n:.3f} | wrote {out_dir}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
        send_bot_message(
            f"GSM8K RJEPA row finished:\n"
            f"Predictor: {Path(args.predictor_checkpoint).name}\n"
            f"Output: logs/benchmark/{BENCHMARK}/"
        )
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"GSM8K RJEPA row failed:\n{err_msg}")
        raise
