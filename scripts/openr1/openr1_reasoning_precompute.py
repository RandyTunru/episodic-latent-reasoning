"""Precompute frozen-encoder representations for R-JEPA training.

Runs a frozen encoder backbone over the open-r1 Mixture-of-Thoughts
"science" split and stores ONE branch of the R-JEPA input contract per
run, selected by ``--branch``:

    targets  - pooled per-step encodings (the R-JEPA regression targets).
               One row per sample: ``(sample_index, num_steps,
               num_steps_stored, step_targets)``, where ``step_targets``
               is an unpadded bf16 blob.  ``num_steps`` is the TRUE
               (uncapped) step count from the separation - the router's
               halt signal, so the column is never capped.
               ``--max-steps`` caps only the stored blob;
               ``num_steps_stored`` records that capped count
               (``min(num_steps, max_steps)`` vectors per row) so the
               training-time bucketing sampler has the exact effective
               length without decoding blobs.
    ctx      - token-level instruction encodings (the predictor's context
               conditioning).  One row per sample: ``(sample_index,
               ctx_len, ctx_embeddings)``, an unpadded ``(ctx_len, E)``
               bf16 blob.  ``ctx_len`` is stored explicitly so training
               can bucket batches without scanning the binary column.

The two branches are independent runs: choose any ``--model-id`` per
branch (e.g. the LLM backbone for ctx, an embedding sibling of it for
targets).  They are joined at load time on ``sample_index`` - row order
is not part of the contract, so each branch can bucket its own shard for
GPU efficiency.

Blobs are raw little-endian bytes; shapes recover via ``len(blob) //
(E * 2)``.  The contract itself lives in
``src/data/helper/precomputed.py`` and is validated against this
script's own output before it reports success.

Distributed: each rank wraps the encoder in DDP and encodes a shard of
the dataset, writing its own part file (no filesystem coordination).
This script only ever runs the encoder FORWARD, so there are no gradient
collectives; the only cross-rank traffic is the process-group handshake
and DDP's one-time parameter broadcast at construction.  If NCCL init
fails, the script retries with gloo (CPU/TCP), which is sufficient for
forward-only work.  If NCCL hangs rather than raising, pass
``--backend gloo`` directly.

Run:
    torchrun --nproc_per_node=2 scripts/openr1/openr1_reasoning_precompute.py \
        --branch targets --model-id Qwen/Qwen3-Embedding-4B
    torchrun --nproc_per_node=2 scripts/openr1/openr1_reasoning_precompute.py \
        --branch ctx --model-id Qwen/Qwen3-4B
"""
from dotenv import load_dotenv
load_dotenv()  # for WANDB_API_KEY in .env

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.helper.collate import pad_collate
from src.data.openr1_reasoning_dataset import ReasoningDataset
from src.data.helper.precomputed import SCHEMAS, validate_parquet


# ----------------------------------------------------------------------
# Distributed setup
# ----------------------------------------------------------------------

def init_process_group(rank: int, world_size: int, backend_arg: str) -> str:
    """Initialize the process group; fall back to gloo when NCCL fails.

    Forward-only work issues no GPU collectives, so gloo (CPU/TCP) is a
    perfectly adequate backend when the GPUs' NCCL path is broken.
    """
    backends = [backend_arg] if backend_arg in ("nccl", "gloo") else ["nccl", "gloo"]
    last_error = None
    for backend in backends:
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(hours=24),
            )
            print(f"[rank {rank}] process group initialized (backend={backend})", flush=True)
            return backend
        except Exception as exc:  # try the next backend
            last_error = exc
            if dist.is_initialized():
                dist.destroy_process_group()
            print(f"[rank {rank}] {backend} init failed: {exc}", flush=True)
    raise RuntimeError(f"could not initialize any process group: {last_error}")


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------

def to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a CPU tensor as raw little-endian bytes (no dtype metadata)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        # numpy's bfloat16 support is spotty; keep the raw bytes instead.
        tensor = tensor.view(torch.uint8)
    return tensor.numpy().tobytes()


def pool_steps(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool token-level encodings into one vector per row.

    Mirrors ``RJEPA._pool`` so precomputed targets match what the online
    path would have produced.

    Args:
        hidden: ``(M, L, E)`` token encodings.
        mask: ``(M, L)`` bool, True = valid token.
        mode: ``"eos"`` (last valid token), ``"mean"``, or ``"cls"``.

    Returns:
        ``(M, E)`` pooled vectors.
    """
    if mode == "eos":
        idx = mask.sum(dim=1).clamp(min=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    if mode == "mean":
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
    if mode == "cls":
        return hidden[:, 0]
    raise ValueError(f"unsupported pooling mode: {mode}")


def bucket_order(data, branch: str, step_separator: str):
    """Sort sample indices to minimize padding waste for this branch.

    ``targets`` sorts by (num steps, longest step) - the two axes the
    collate pads.  ``ctx`` sorts by instruction length.  Rows need not be
    ordered the same across branches: the training loader joins on
    ``sample_index``, so each branch optimizes its own batching.
    """
    if branch == "ctx":
        keys = [len(row["messages"][0]["content"]) for row in data]
    else:
        pattern = re.compile(re.escape("<think>") + r"(.*?)" + re.escape("</think>"), re.DOTALL)
        keys = []
        for row in data:
            response = row["messages"][1]["content"]
            match = pattern.search(response)
            cot = match.group(1).strip() if match else response.strip()
            steps = [s.strip() for s in re.split(step_separator, cot) if s.strip()]
            # Primary: number of steps (steps-axis padding).  Secondary:
            # longest step (token-axis padding - the collate pads every
            # step in the batch to the batch's longest step).
            keys.append((len(steps), max(len(s) for s in steps) if steps else 0))
    return sorted(range(len(keys)), key=lambda i: keys[i])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch", required=True, choices=sorted(SCHEMAS),
        help="which side of the R-JEPA input contract to produce",
    )
    parser.add_argument("--dataset-id", default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--subset", default="science")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-id", required=True, help="encoder backbone to run")
    parser.add_argument(
        "--pooling", default="eos", choices=["eos", "mean", "cls"],
        help="targets branch only: how to pool each step's tokens",
    )
    parser.add_argument(
        "--step-separator", default=r"\n\n+",
        help="targets branch only: regex that splits CoT into steps",
    )
    parser.add_argument(
        "--max-ctx-tokens", type=int, default=None,
        help="ctx branch only: instruction truncation length "
        "(None = store the full instruction)" 
        "(discouraged - cap at load via the training config instead)",
    )
    parser.add_argument(
        "--max-step-tokens", type=int, default=None,
        help="targets branch only: per-step truncation length "
        "(None = store each full step)"
        "(discouraged - cap at load via the training config instead)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="targets branch only: cap the STORED step_targets blob per "
        "sample (None = store every step).  num_steps still records the "
        "TRUE step count - it is the router's halt signal, never capped.  "
        "num_steps_stored records the capped count so training buckets "
        "by the exact effective length",
    )
    parser.add_argument(
        "--step-tokens-per-chunk", type=int, default=65536,
        help="targets branch only: cap tokens per encoder call on flattened steps",
    )
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers so per-item tokenization overlaps GPU encoding",
    )
    
    parser.add_argument("--backend", default="auto", choices=["auto", "nccl", "gloo"])
    parser.add_argument("--max-samples", type=int, default=None, help="debug: process only the first N samples")
    parser.add_argument(
        "--only-rank", type=int, default=None,
        help="encode only the round-robin share normally assigned to this "
        "rank (partial re-runs; pair with --nproc_per_node=1 and "
        "--num-ranks equal to the original launch)",
    )
    parser.add_argument(
        "--num-ranks", type=int, default=None,
        help="total shards the round-robin was split across "
        "(defaults to WORLD_SIZE; only needed with --only-rank)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "openr1_science_precompute"),
        help="part files are written to <output_dir>/<branch>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    backend = None
    if distributed:
        backend = init_process_group(rank, world_size, args.backend)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- dataset: tokenized per-sample, sharded per rank ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    data = load_dataset(args.dataset_id, args.subset, split=args.split)
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))

    dataset = ReasoningDataset(
        data,
        tokenizer,
        max_ctx_token_length=args.max_ctx_tokens,
        max_step_token_length=args.max_step_tokens,
        step_separator=args.step_separator,
    )

    # Length-bucketed, round-robin shard per rank.  Batches are formed from
    # contiguous runs of the sorted order (homogeneous lengths -> low
    # padding waste) and then distributed round-robin so every rank gets
    # equal shares of the short and long ends (balanced wall time).
    total = len(dataset)
    order = bucket_order(data, args.branch, args.step_separator)
    batches = [order[i : i + args.batch_size] for i in range(0, total, args.batch_size)]
    n_shards = args.num_ranks if args.num_ranks is not None else world_size
    share_rank = rank if args.only_rank is None else args.only_rank
    my_indices = [idx for b in batches[share_rank::n_shards] for idx in b]
    shard = Subset(dataset, my_indices)

    loader = DataLoader(
        shard, batch_size=args.batch_size, collate_fn=pad_collate,
        shuffle=False, num_workers=args.num_workers,
    )

    # ---- frozen encoder (forward-only; DDP wrapper for multi-GPU) ----
    encoder = AutoModel.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    if distributed:
        encoder = DDP(encoder, device_ids=[rank])
    encoder.eval()

    base_encoder = encoder.module if distributed else encoder
    E = base_encoder.config.hidden_size

    out_dir = Path(args.output_dir) / args.branch / args.model_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    part_file = out_dir / f"part-{share_rank:03d}.parquet"
    part_file.unlink(missing_ok=True)  # ParquetWriter appends to existing files

    schema = SCHEMAS[args.branch]
    # Small row groups keep random access cheap at load time: the dataset
    # reader fetches single rows via take(), which decodes the containing
    # row group, so a giant default group would mean decoding the whole
    # file per sample.  32 rows ≈ 4-10 MB per group.
    writer = pq.ParquetWriter(part_file, schema)
    ROW_GROUP_SIZE = 32

    rows = []
    ctx_bytes_total, step_bytes_total = 0, 0
    ctx_tokens_total, step_tokens_total = 0, 0
    write_every = 256
    row_offset = 0  # running position in my_indices (robust to partial batches)
    t0 = time.time()

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"[rank {rank}] {args.branch}", disable=rank != 0)):
            B = batch["ctx_input_ids"].size(0)
            ctx_ids = batch["ctx_input_ids"].to(device)
            ctx_mask = batch["ctx_attention_mask"].to(device)

            if args.branch == "ctx":
                # --- context branch: token-level encodings (unpadded on write) ---
                ctx_hidden = encoder(
                    input_ids=ctx_ids, attention_mask=ctx_mask,
                ).last_hidden_state  # (B, C, E)
                ctx_tokens_total += ctx_ids.numel()
            else:
                # --- targets branch: flatten -> encode -> pool ---
                step_ids = batch["step_input_ids"].to(device)
                step_mask = batch["step_attention_mask"].to(device)
                num_steps = batch["num_steps"]
                B, S, L = step_ids.shape

                if num_steps.max().item() > 0:
                    flat_ids = step_ids.reshape(B * S, L)
                    flat_mask = step_mask.reshape(B * S, L)
                    chunk_rows = max(1, args.step_tokens_per_chunk // max(L, 1))
                    pooled = torch.zeros(B * S, E, dtype=torch.bfloat16, device=device)
                    for lo in range(0, B * S, chunk_rows):
                        hi = min(B * S, lo + chunk_rows)
                        h = encoder(
                            input_ids=flat_ids[lo:hi], attention_mask=flat_mask[lo:hi],
                        ).last_hidden_state  # (chunk, L, E)
                        pooled[lo:hi] = pool_steps(h, flat_mask[lo:hi], args.pooling)
                        step_tokens_total += (hi - lo) * L
                    pooled = pooled.reshape(B, S, E)
                else:
                    pooled = torch.zeros(B, S, E, dtype=torch.bfloat16, device=device)

            # --- slice to unpadded per-sample rows ---
            for i in range(B):
                sample_index = my_indices[row_offset + i]  # running offset: robust to partial batches
                if args.branch == "ctx":
                    ctx_len = int(ctx_mask[i].sum().item())
                    rows.append({
                        "sample_index": sample_index,
                        "ctx_len": ctx_len,
                        "ctx_embeddings": to_bytes(ctx_hidden[i, :ctx_len]),
                    })
                    ctx_bytes_total += ctx_len * E * 2
                else:
                    n_true = int(num_steps[i].item())
                    n_store = n_true
                    if args.max_steps is not None:
                        n_store = min(n_store, args.max_steps)
                    rows.append({
                        "sample_index": sample_index,
                        "num_steps": n_true, # the true step count (uncapped)
                        "num_steps_stored": n_store, # the capped step count (capped by --max-steps)
                        "step_targets": to_bytes(pooled[i, :n_store]),
                    })
                    step_bytes_total += n_store * E * 2
            row_offset += B

            if len(rows) >= write_every:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=ROW_GROUP_SIZE)
                rows = []

    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    writer.close()

    # ---- self-validate against the shared contract ----
    n_rows = validate_parquet(part_file, args.branch, E)
    if n_rows != len(my_indices):
        raise ValueError(
            f"{part_file}: expected {len(my_indices)} rows, found {n_rows}"
        )

    elapsed = time.time() - t0
    stats = {
        "branch": args.branch,
        "rank": rank,
        "backend": backend,
        "samples": len(my_indices),
        "ctx_bytes": ctx_bytes_total,
        "step_bytes": step_bytes_total,
        "encoded_tokens": ctx_tokens_total + step_tokens_total,
        "elapsed_s": elapsed,
        "model_id": args.model_id,
        "pooling": args.pooling,
        "hidden_size": E,
        "dtype": "bfloat16",
        "part_file": str(part_file),
    }
    (out_dir / f"part-{share_rank:03d}.json").write_text(json.dumps(stats, indent=2))
    branch_bytes = ctx_bytes_total + step_bytes_total
    tps = stats["encoded_tokens"] / max(elapsed, 1e-6)
    print(
        f"[rank {rank}] wrote {len(my_indices)} samples -> {part_file} "
        f"({branch_bytes / 1e9:.2f} GB in {elapsed:.0f}s, {tps:,.0f} tokens/s) "
        f"[validated]",
        flush=True,
    )

    if distributed and args.only_rank is None:
        dist.barrier()
        if rank == 0:
            parts = [
                json.loads((out_dir / f"part-{r:03d}.json").read_text())
                for r in range(world_size)
            ]
            summary = {
                "branch": args.branch,
                "dataset": f"{args.dataset_id}/{args.subset}/{args.split}",
                "model_id": args.model_id,
                "pooling": args.pooling,
                "hidden_size": E,
                "dtype": "bfloat16",
                "total_samples": sum(p["samples"] for p in parts),
                "ctx_bytes": sum(p["ctx_bytes"] for p in parts),
                "step_bytes": sum(p["step_bytes"] for p in parts),
                "encoded_tokens": sum(p["encoded_tokens"] for p in parts),
                "elapsed_s": sum(p["elapsed_s"] for p in parts),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(
                f"done: {summary['total_samples']} samples "
                f"({(summary['ctx_bytes'] + summary['step_bytes']) / 1e9:.2f} GB) "
                f"-> {out_dir}",
                flush=True,
            )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
