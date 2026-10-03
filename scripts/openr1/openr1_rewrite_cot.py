"""Rewrite openr1 chain-of-thoughts into compact sequential reasoning steps.

Reads a HuggingFace dataset, extracts each row's CoT from the model
response (between ``<think></think>`` tags - same extraction rule as
``ReasoningDataset._extract_cot``), and asks a vLLM-served LLM to
rewrite it into a clean, step-separated (``\\n\\n``) progression.

ONE ROW PER (SAMPLE, REWRITE VARIANT).  Each output row copies the
original dataset columns and adds:

    orig_row_id  int64   - original dataset row index; all variants of
                           one sample share it
    cot_variant  int32   - 0 .. n_rewrite-1
    cot_text     string  - the rewritten CoT ("" when the original had
                           no CoT or the rewrite failed - rows are never
                           dropped, so the join never desynchronizes)

Output: ``<output_dir>/part-XXX.parquet`` - contiguous row-count shards
(each part later translates 1-to-1 into a ``.arrow`` file) - plus
``<output_dir>/meta.json`` carrying ``n_rewrite`` and the exact
generation settings (prompt, model, temperature, seed base): the
rewritten text is a data layer and must be reproducible.

The precompute script consumes this directory with
``--cot-source rewritten`` and derives the join key as
``sample_index = orig_row_id * n_rewrite + cot_variant`` (both
branches); training recovers the variant group as
``sample_index // n_rewrite`` for the group-aware val holdout.

Single process by design: one global asyncio semaphore throttles the
vLLM server (per-worker-process semaphores would multiply the request
rate), and row-count sharding gives the small part files that parallel
transfers want without multiprocessing coordination.

Run:
    python scripts/openr1/openr1_rewrite_cot.py \
        --subset science --n-rewrite 3 \
        --output-dir data/openr1_science_rewritten_cot
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.telegram_service import send_bot_message
from tools.on_error import format_error
from tools.vllm_client import VLLMClient

MODEL = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.8-27B")
TEMPERATURE = float(os.environ.get("VLLM_TEMPERATURE", 0.8))
MAX_CONCURRENT = int(os.environ.get("VLLM_SERVER_CONCURRENT_REQUESTS", 5))
MAX_RETRIES = int(os.environ.get("VLLM_MAX_RETRIES", 3))
RETRY_DELAY_S = float(os.environ.get("VLLM_RETRY_DELAY_S", 5.0))
SEED_BASE = int(os.environ.get("VLLM_SEED", 42))
CLOSE_THINK_TOKEN = os.environ.get("VLLM_THINKING_CLOSE_TOKEN", "</think>")

# One client over every server URL in VLLM_BASE_URL (comma/space-separated
# or a JSON list): requests take the first free slot across servers, and a
# failing server cools down for 30s.  max_concurrent is PER SERVER, so the
# total concurrent request budget is max_concurrent x n_servers.
CLIENT = VLLMClient(
    base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000"),
    model=MODEL,
    api_key=os.environ.get("VLLM_API_KEY", None),
    max_concurrent=MAX_CONCURRENT,
    timeout=600.0,
    retries=MAX_RETRIES,
)

REWRITE_PROMPT = """### Instruction:
You are given a chain-of-thought (CoT) reasoning text extracted from a model response. Rewrite it into a compact, ordered sequence of discrete reasoning steps.

### Instructions:
1. Understand the original CoT and its reasoning flow.
2. Each step should be a complete sentence, clearly stating a single reasoning point.
3. No need to preserve the original wording; focus on clarity and conciseness.
4. Do not include any meta-thoughts or commentary; only the reasoning steps matter.
5. Ensure you follow the dynamics of the original reasoning, if it involves a change of viewpoint, a new assumption, or a new line of reasoning, reflect that in the step sequence.
6. For steps that involves sequential steps ensure that the order is preserved. 

### Special Case and Formatting Rules:
Never write numbered steps or bullet points as a single list within a single paragraph. 
Here is an example of what NOT to do:
"
To implement this:
1. Split the input string into $S$ and $T$.
2. Construct the string $T + \# + S$.
3. Compute the prefix function for this concatenated string.
4. Initialize a DP array $dp$ of size $m+1$ with $dp[0] = 1$.
5. For each $i$ from 1 to $m$:
   - Retrieve the prefix function value at the corresponding position in the concatenated string.
   - Traverse the failure function chain to find all valid $l$.
   - For each valid $l$, update $dp[i] += dp[i-l]$.
6. Output $dp[m]$.
"
Instead, rewrite each component as its own separate step with a double newline character ("\\n\\n").

### Rules:
1. Preserve the original reasoning steps and conclusions. Do not fix, correct, or improve the reasoning itself.
2. Remove verbosity, repetition, rambling, and filler. Merge steps that restate the same operation.
3. Avoid introducing new information or changing the original meaning.
4. Use clear and concise language, and ensure that the reasoning is easy to follow.
5. Ensure that the rewritten CoT is logically structured and maintains the original intent of the reasoning.
6. Ensure that every step retains its original detail and information, but is expressed in a clearer and more concise manner.
7. Separate each reasoning step with a double newline character ("\\n\\n"). Output ONLY the rewritten reasoning - no headers, no commentary, no quotes.
8. Ensure each rewrite ends with a final conclusion or answer, if present in the original CoT.
"""


VARIANT_LINE = (
    "9. This is variant {variant} of {n} independent rewrites of the same excerpt. "
    "Choose a different grouping, ordering emphasis, or compression of the steps "
    "than the other variants."
)

THINK_RE = re.compile(
    re.escape("<think>") + r"(.*?)" + re.escape("</think>"), re.DOTALL,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="open-r1/Mixture-of-Thoughts")
    parser.add_argument("--subset", default="science")
    parser.add_argument("--split", default="train")

    parser.add_argument(
        "--thinking", action="store_true",
        help="run the rewriter with vLLM thinking enabled (the rewrite is "
        "taken from after the closing think tag; meta-thoughts are discarded)",
    )
    parser.add_argument(
        "--n-rewrite", type=int, default=1,
        help="number of rewrite variants per sample (each variant becomes "
        "its own row with the same orig_row_id)",
    )
    parser.add_argument(
        "--rows-per-part", type=int, default=100_000,
        help="row-count shard size: each part file holds a contiguous run "
        "of this many rows (translates 1-to-1 into a .arrow file later)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="rows scheduled per async gather (the semaphore caps actual "
        "concurrent requests against the vLLM server)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="debug: process only the first N samples of the dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "openr1_science_rewritten_cot"),
    )
    return parser.parse_args()


def _extract_cot(response: str) -> str:
    """Extract the CoT from a model response (same rule as
    ReasoningDataset._extract_cot: text between the think tags, or the
    whole response when the tags are missing)."""
    match = THINK_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def _extract_rewrite(content: str) -> str:
    """Keep the text AFTER the last closing think tag - the rewriter's
    final output, with any meta-thoughts discarded.  With thinking
    disabled there are no tags and the whole content is the rewrite."""
    return content.split(CLOSE_THINK_TOKEN)[-1].strip()


def _prompt(n_rewrite: int, variant: int) -> str:
    if n_rewrite > 1:
        return REWRITE_PROMPT + "\n\n" + VARIANT_LINE.format(
            variant=variant + 1, n=n_rewrite,
        )
    return REWRITE_PROMPT


async def _rewrite_row(
    orig_row_id: int,
    variant: int,
    cot_text: str,
    row: dict,
    n_rewrite: int,
    thinking: bool,
    stats: dict,
    failures,
) -> dict:
    """Build one output row.  Rows are never dropped: a failed call (or
    an empty original CoT) emits cot_text="" so the row count stays
    exactly n_original * n_rewrite."""
    out_row = dict(row)
    out_row["orig_row_id"] = orig_row_id
    out_row["cot_variant"] = variant
    if not cot_text:
        stats["empty_originals"] += 1
        out_row["cot_text"] = ""
        return out_row
    try:
        response = await CLIENT.text_completion(
            messages=[
                {"role": "system", "content": _prompt(n_rewrite, variant)},
                {"role": "user", "content": cot_text},
            ],
            max_tokens=20480,
            thinking=thinking,
            temperature=TEMPERATURE,
            seed=SEED_BASE + orig_row_id * n_rewrite + variant,
        )

        # With thinking enabled the response still contains the rewriter's
        # <think> block; keep only what follows the closing tag.
        out_row["cot_text"] = _extract_rewrite(response)
    except Exception as exc:
        stats["failures"] += 1
        out_row["cot_text"] = ""
        failures.write(
            json.dumps({
                "orig_row_id": orig_row_id,
                "cot_variant": variant,
                "error": str(exc),
            }) + "\n"
        )
    return out_row


def _infer_schema(data) -> pa.Schema:
    """Copied original columns keep their inferred types; the columns the
    rewrite owns are declared explicitly.  Type conflicts later surface
    as write-time errors instead of silent coercion."""
    probe = [{k: data[i][k] for k in data.column_names}
             for i in range(min(100, len(data)))] # sample 100 rows to infer the original schema
    copied_schema = pa.Table.from_pylist(probe).schema
    owned_schema = pa.schema([
        ("orig_row_id", pa.int64()),
        ("cot_variant", pa.int32()),
        ("cot_text", pa.string()),
    ])
    return pa.unify_schemas([copied_schema, owned_schema])


async def main(args: argparse.Namespace) -> None:
    data = load_dataset(args.dataset_id, args.subset, split=args.split)
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))
    n_original = len(data)
    if n_original == 0:
        raise ValueError("empty dataset")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = _infer_schema(data)
    failures_path = out_dir / "failures.jsonl"
    failures_path.unlink(missing_ok=True)

    # Every configured server must be reachable and serve the same model
    # before any rewrite is spent (this also catches a wrong second URL).
    await CLIENT.check_models()

    stats = {"failures": 0, "empty_originals": 0}
    t0 = time.time()

    # Row-count sharded writers: each part holds a contiguous run of
    # rows, closed and rotated at the shard boundary.
    part_idx, part_rows, written = 0, 0, 0
    writer = pq.ParquetWriter(out_dir / f"part-{part_idx:03d}.parquet", schema)
    pending = []

    def flush() -> None:
        nonlocal pending, part_idx, part_rows, writer, written
        while pending:
            take = min(len(pending), args.rows_per_part - part_rows)
            chunk, pending = pending[:take], pending[take:]
            writer.write_table(
                pa.Table.from_pylist(chunk, schema=schema), row_group_size=1024,
            )
            written += take
            part_rows += take
            if part_rows >= args.rows_per_part:
                writer.close()
                part_idx += 1
                writer = pq.ParquetWriter(
                    out_dir / f"part-{part_idx:03d}.parquet", schema,
                )
                part_rows = 0

    n_batches = (n_original + args.batch_size - 1) // args.batch_size
    with open(failures_path, "a") as failures:
        for batch_idx, batch in enumerate(
            tqdm(data.iter(batch_size=args.batch_size), total=n_batches)
        ):
            batch_start = batch_idx * args.batch_size
            jobs = []
            for j in range(len(batch["messages"])):
                orig_row_id = batch_start + j
                row = {k: batch[k][j] for k in batch}
                cot_text = _extract_cot(row["messages"][1]["content"])
                for variant in range(args.n_rewrite):
                    jobs.append(_rewrite_row(
                        orig_row_id, variant,
                        cot_text, row, args.n_rewrite, args.thinking,
                        stats, failures,
                    ))
            pending.extend(await asyncio.gather(*jobs))
            flush()

    flush()
    writer.close()
    await CLIENT.aclose()

    elapsed = time.time() - t0
    meta = {
        "dataset_id": args.dataset_id,
        "subset": args.subset,
        "split": args.split,
        "model": MODEL,
        "base_urls": [ep.url for ep in CLIENT.endpoints],
        "temperature": TEMPERATURE,
        "thinking": args.thinking,
        "n_rewrite": args.n_rewrite,
        "seed_base": SEED_BASE,
        "max_concurrent": MAX_CONCURRENT,
        "max_retries": MAX_RETRIES,
        "rows_per_part": args.rows_per_part,
        "original_rows": n_original,
        "total_rows": n_original * args.n_rewrite,
        "empty_cot_rows": stats["empty_originals"],
        "failed_rows": stats["failures"],
        "prompt": REWRITE_PROMPT,
        "variant_line": VARIANT_LINE,
        "columns": schema.names,
        "num_parts": part_idx + 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"wrote {written} rows ({n_original} samples x {args.n_rewrite} "
        f"variants) -> {out_dir} in {elapsed:.0f}s "
        f"[failures={stats['failures']}, empty_originals={stats['empty_originals']}]",
        flush=True,
    )


if __name__ == "__main__":
    args = parse_args()

    try:
        asyncio.run(main(args))
        send_bot_message(
            "Rewrite CoT finished: \n"
            f"Dataset: {args.dataset_id}/{args.subset}/{args.split}\n"
            f"Model: {MODEL}, n_rewrite: {args.n_rewrite}\n"
            f"Output: {args.output_dir}"
        )
    except Exception as e:
        err_msg = format_error(e)
        send_bot_message(
            "Rewrite CoT failed: \n"
            f"Dataset: {args.dataset_id}/{args.subset}/{args.split}\n"
            f"Model: {MODEL}\n\n"
            f"Details:\n{err_msg}"
        )
        raise
