"""Generate per-row latent reasoning chains with a trained predictor
checkpoint and store them as a precomputed branch (like ctx/targets).

Each row's instruction ctx embeddings come from the precomputed ctx
branch; the predictor runs FREE (no teacher forcing): pack
``[ctx | START]``, forecast step 0, check the router, append and repeat
until the router halts or the generation cap is reached.  The stored
chain INCLUDES the halting forecast (the conclusion latent the LLM
consumes last).

Distributed: same convention as ``openr1_reasoning_precompute.py`` -
launch with torchrun, rows are batched in key order and the batches are
distributed round-robin across ranks, each rank writes its own
``part-{rank:03d}.parquet`` + ``part-{rank:03d}.json`` sidecar.  No DDP:
this is pure inference (no gradients, no collectives).  Generation is
deterministic (no sampling), so a rank's part is reproducible; partial
re-runs use ``--only-rank R --num-ranks N`` with nproc=1.

Schema (part files):
    sample_index  int64    - join key into the ctx/targets branches
    num_steps     int32    - generated chain length (incl. conclusion)
    halted        int8     - 1 = router halted, 0 = hit the generation cap
    step_latents  binary   - (num_steps, encoder_dim) bf16 blob
    router_logits binary   - (num_steps,) float32 blob, kept so the halt
                             threshold can be swept post-hoc without
                             regenerating

Run:
    torchrun --nproc_per_node=2 scripts/openr1/openr1_latent_precompute.py \
        --config config/openr1_all/qwen3_4b_rewritten_rjepa.yaml \
        --checkpoint artifacts/checkpoints/<run>/model_step_XXXXX.pt \
        --output-dir data/processed/openr1_all_latent_precomputed/<run>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.openr1.data.openr1_reasoning_precompute import init_process_group
from src.data.precomputed_dataset import PrecomputedReasoningDataset
from src.models.r_jepa.r_jepa import CausalRJEPA
from tools.telegram_service import send_bot_message
from tools.on_error import format_error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="predictor config (defines the model + ctx/targets dirs)")
    p.add_argument("--checkpoint", required=True,
                   help="predictor checkpoint .pt (model_state_dict)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64,
                   help="generation batch (rows processed in key order)")
    p.add_argument("--max-steps", type=int, default=64,
                   help="generation cap; chains longer than this are stored halted=0")
    p.add_argument("--halt-threshold", type=float, default=0.0,
                   help="router halt rule: first step with logit > tau")
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug: limit the global rows considered (pre-shard)")
    p.add_argument("--backend", type=str, default="nccl",
                   help="process-group backend (nccl with gloo fallback)")
    p.add_argument("--only-rank", type=int, default=None,
                   help="generate only the round-robin share normally assigned "
                        "to this rank (partial re-runs; pair with nproc=1 and "
                        "--num-ranks equal to the original launch)")
    p.add_argument("--num-ranks", type=int, default=None,
                   help="total shards the round-robin was split across "
                        "(defaults to WORLD_SIZE; only needed with --only-rank)")
    return p.parse_args()


SCHEMA = pa.schema([
    ("sample_index", pa.int64()),
    ("num_steps", pa.int32()),
    ("halted", pa.int8()),
    ("step_latents", pa.binary()),
    ("router_logits", pa.binary()),
])

def to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a CPU tensor as raw little-endian bytes (no dtype metadata)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        # numpy's bfloat16 support is spotty; keep the raw bytes instead.
        tensor = tensor.view(torch.uint8)
    return tensor.numpy().tobytes()

def generate_batch(predictor, ctxs, device, tau, max_steps):
    """Free-running generation for one batch of ctx tensors.

    Returns per-row (chain (S, E) bf16, logits (S,) f32, halted bool).
    """
    B = len(ctxs)
    E = ctxs[0].size(-1)
    steps = [[] for _ in range(B)]
    done = [False] * B
    logits = [[] for _ in range(B)]

    for _ in range(max_steps):
        if all(done):
            break
        lens = [ctxs[i].size(0) + len(steps[i]) for i in range(B)]
        T = max(lens)
        packed = torch.zeros(B, T, E, dtype=torch.bfloat16, device=device)
        valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i in range(B):
            L = lens[i]
            packed[i, :ctxs[i].size(0)] = ctxs[i].to(device)
            for k, s in enumerate(steps[i]):
                packed[i, ctxs[i].size(0) + k] = s
            valid[i, :L] = True
        ctx_len = torch.tensor([c.size(0) for c in ctxs], device=device)
        S_out = max(len(s) for s in steps) + 1

        with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16):
            preds, rlogits = predictor.forward_packed(
                packed, valid, ctx_len, S_out,
            )

        for i in range(B):
            if done[i]:
                continue
            j = len(steps[i])
            steps[i].append(preds[i, j])
            logits[i].append(rlogits[i, j].item())
            if rlogits[i, j].item() > tau:
                done[i] = True

    out = []
    for i in range(B):
        # numpy cannot serialize bf16: view the raw bytes as uint8 and
        # tobytes() that - the same layout torch.frombuffer(dtype=bf16)
        # reconstructs on the read side.
        chain = torch.stack(steps[i]).contiguous().to("cpu")
        out.append((
            to_bytes(chain),
            np.asarray(logits[i], dtype=np.float32).tobytes(),
            int(done[i]),
        ))
    return out


def _blob_len(blob: bytes, hidden: int) -> int:
    return len(blob) // (hidden * 2)


def main(args) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = init_process_group(rank, world_size, args.backend)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open(args.config))

    dataset = PrecomputedReasoningDataset(
        ctx_dir=config["ctx_dir"], targets_dir=config["targets_dir"],
        hidden_size=config["encoder_hidden_size"],
        max_steps=config["predictor_max_seq_len"],
        max_ctx_len=config["encoder_max_seq_len"],
        validate=False,
    )
    n_rows = len(dataset) if args.max_samples is None else args.max_samples

    predictor_kwargs = dict(
        encoder_dim=config["encoder_hidden_size"],
        predictor_dim=config["predictor_hidden_size"],
        num_heads=config["predictor_num_heads"],
        d_ff=config["predictor_d_ff"],
        num_layers=config["predictor_num_layers"],
        max_seq_length=config["predictor_max_seq_len"],
        dropout=0.0,
    )
    model = CausalRJEPA(
        predictor_kwargs=predictor_kwargs, encoder=None,
        encoder_max_seq_length=config["encoder_max_seq_len"],
        encoder_pooling=config["encoder_pooling"],
    ).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    predictor = model.predictor
    if rank == 0:
        print(f"loaded {args.checkpoint}", flush=True)

    # ---- round-robin share: batches in key order, rank takes r::n_shards ----
    n_shards = args.num_ranks if args.num_ranks is not None else world_size
    share_rank = rank if args.only_rank is None else args.only_rank
    batch_starts = list(range(0, n_rows, args.batch_size))
    my_batches = [
        (b, min(b + args.batch_size, n_rows))
        for b in batch_starts[share_rank::n_shards]
    ]
    print(f"[rank {rank}] rows: {sum(e - b for b, e in my_batches)} across "
          f"{len(my_batches)} batches", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_file = out_dir / f"part-{share_rank:03d}.parquet"
    writer = pq.ParquetWriter(part_file, SCHEMA)

    stats = {"halted": 0, "capped": 0, "lengths": [], "rows": 0}
    t0 = time.time()

    for bi, (b, e) in enumerate(my_batches):
        rows = [dataset[i] for i in range(b, e)]
        ctxs = [r["ctx_embeddings"] for r in rows]
        chains = generate_batch(
            predictor, ctxs, device, args.halt_threshold, args.max_steps,
        )
        for i, (blob_lat, blob_log, halted) in enumerate(chains):
            stats["halted" if halted else "capped"] += 1
            stats["lengths"].append(_blob_len(blob_lat, config["encoder_hidden_size"]))
            writer.write_table(pa.Table.from_pylist([{
                "sample_index": dataset.keys[b + i],
                "num_steps": _blob_len(blob_lat, config["encoder_hidden_size"]),
                "halted": halted,
                "step_latents": blob_lat,
                "router_logits": blob_log,
            }], schema=SCHEMA), row_group_size=1024)
            stats["rows"] += 1
        if bi % 10 == 0:
            print(
                f"[rank {rank}] rows {stats['rows']} | "
                f"halted={stats['halted']} capped={stats['capped']} | "
                f"{time.time() - t0:.0f}s", flush=True,
            )

    writer.close()
    lengths = np.asarray(stats["lengths"])
    sidecar = {
        "branch": "latents",
        "rank": share_rank,
        "num_ranks": n_shards,
        "checkpoint": str(args.checkpoint),
        "halt_threshold": args.halt_threshold,
        "max_steps": args.max_steps,
        "rows": stats["rows"],
        "halted": stats["halted"],
        "capped": stats["capped"],
        "chain_len_mean": float(lengths.mean()),
        "chain_len_median": float(np.median(lengths)),
        "chain_len_p90": float(np.percentile(lengths, 90)),
        "chain_len_max": int(lengths.max()),
        "elapsed_s": time.time() - t0,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"part-{share_rank:03d}.json").write_text(
        json.dumps(sidecar, indent=2),
    )
    print(f"[rank {rank}] done: {json.dumps(sidecar, indent=2)}", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
        send_bot_message(
            f"Latent precompute finished (rank "
            f"{os.environ.get('LOCAL_RANK', '0')}):\n"
            f"Checkpoint: {args.checkpoint}\nOutput: {args.output_dir}"
        )
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(f"Latent precompute failed:\n{err_msg}")
        raise
