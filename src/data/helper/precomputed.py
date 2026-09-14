"""Schema + validation contract for precomputed R-JEPA representation files.

Each precompute run with ``--branch`` produces one parquet file per rank
under ``<output_dir>/<branch>/``:

    targets:  sample_index      int64  - join key, original dataset index
              num_steps         int32  - TRUE number of reasoning steps
                                          (never capped - the router's
                                          halt signal)
              num_steps_stored  int32  - vectors actually stored in the
                                          blob, ``min(num_steps,
                                          max_steps)`` (the bucketing
                                          key at load time)
              step_targets      binary - pooled per-step vectors,
                                          unpadded ``(num_steps_stored,
                                          E)`` bf16

    ctx:      sample_index   int64  - join key, original dataset index
              ctx_len        int32  - number of tokens in the instruction
              ctx_embeddings binary - token-level instruction encodings,
                                      unpadded ``(ctx_len, E)`` bf16

Blobs are raw little-endian bytes (no dtype/shape metadata inside); the
first axis of each blob is recoverable as ``len(blob) // (E * 2)``, and
the explicit length column (``num_steps`` / ``ctx_len``) makes both
lengths available eagerly - the training-time bucketing sampler needs
them without scanning the binary columns.  The two branches are produced
by separate runs (potentially with different encoder models) and are
joined at load time on ``sample_index`` - row order is *not* part of
the contract.

The training loader validates each file with `validate_parquet`
before consuming it; the precompute script runs the same check on its
own output, so the writer and the reader share one definition of
"well-formed".
"""

from pathlib import Path
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq

BF16_BYTES = 2

SCHEMAS = {
    "targets": pa.schema([
        pa.field("sample_index", pa.int64()),
        pa.field("num_steps", pa.int32()),
        pa.field("num_steps_stored", pa.int32()),
        pa.field("step_targets", pa.binary()),
    ]),
    "ctx": pa.schema([
        pa.field("sample_index", pa.int64()),
        pa.field("ctx_len", pa.int32()),
        pa.field("ctx_embeddings", pa.binary()),
    ]),
}

# Per-branch length column: pairs the explicit length with its blob so
# validation can check the two agree.  targets: ``num_steps`` records
# the true step count and the blob may be shorter when a storage cap
# was used (exact agreement is checked against ``num_steps_stored``);
# ctx: the truncation happens at tokenization, so the column always
# equals the blob.
LENGTH_COLUMNS = {
    "targets": ("num_steps", "step_targets"),
    "ctx": ("ctx_len", "ctx_embeddings"),
}

BRANCHES = tuple(SCHEMAS)


def validate_parquet(
    path: Union[str, Path],
    branch: str,
    hidden_size: int,
    max_rows_to_check: int = 512,
) -> int:
    """Validate a precomputed part file against the branch contract.

    Checks:
    1. Schema matches ``SCHEMAS[branch]`` exactly (names + types).
    2. ``sample_index`` is unique.
    3. Blob byte lengths are divisible by ``hidden_size * BF16_BYTES``
       (checked on the first ``max_rows_to_check`` rows - reading the
       full binary column costs a whole-file scan, and the training
       loader re-derives every blob length when it decodes).
    4. The explicit length columns agree with the blob's element count:
       ``ctx`` - ``ctx_len`` equals the blob rows (the truncation
       happens at tokenization); ``targets`` - ``num_steps_stored``
       equals the blob rows exactly and never exceeds ``num_steps``
       (the true count may exceed what a storage cap stored, never the
       reverse).

    Raises ``ValueError`` on the first violated check.

    Returns:
        Total number of rows in the file.
    """
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch '{branch}' (expected one of {BRANCHES})")

    expected = SCHEMAS[branch]
    parquet = pq.ParquetFile(path)

    # 1. Schema (reads only the footer - cheap).
    if not parquet.schema_arrow.equals(expected):
        raise ValueError(
            f"{path}: schema mismatch\n  got:      {parquet.schema_arrow}\n  expected: {expected}"
        )

    # 2. Unique join keys (the int column is small; read it fully).
    indices = pq.read_table(path, columns=["sample_index"]).to_pandas()["sample_index"]
    if indices.isnull().any():
        raise ValueError(f"{path}: null sample_index found")
    if indices.duplicated().any():
        raise ValueError(f"{path}: duplicate sample_index values found")

    # 3/4. Blob shape consistency (sampled rows only - see docstring).
    vec_bytes = hidden_size * BF16_BYTES
    length_col, blob_col = LENGTH_COLUMNS[branch]
    stored_col = "num_steps_stored" if branch == "targets" else None
    cols = [length_col, blob_col] + ([stored_col] if stored_col else [])
    for batch in parquet.iter_batches(columns=cols, batch_size=256):
        rows = batch.to_pydict()
        for blob in rows[blob_col]:
            if len(blob) % vec_bytes != 0:
                raise ValueError(
                    f"{path}: blob length {len(blob)} not divisible by {vec_bytes} "
                    f"(hidden_size={hidden_size}, bf16)"
                )
        for i in range(len(rows[blob_col])):
            length = rows[length_col][i]
            n_vectors = len(rows[blob_col][i]) // vec_bytes
            if stored_col:
                # The stored count is explicit: the blob must match it
                # exactly, and it can never exceed the true step count.
                stored = rows[stored_col][i]
                if n_vectors != stored:
                    raise ValueError(
                        f"{path}: num_steps_stored={stored} but blob holds "
                        f"{n_vectors} vectors"
                    )
                if stored > length:
                    raise ValueError(
                        f"{path}: num_steps_stored={stored} exceeds "
                        f"num_steps={length}"
                    )
            elif length != n_vectors:
                raise ValueError(
                    f"{path}: {length_col}={length} but blob holds "
                    f"{n_vectors} vectors"
                )
        max_rows_to_check -= len(rows[blob_col])
        if max_rows_to_check <= 0:
            break

    return parquet.metadata.num_rows


def validate_pair(
    ctx_paths,
    target_paths,
    hidden_size: int,
    max_rows_to_check: int = 512,
) -> tuple[int, int]:
    """Validate both branches and confirm they form a consistent pair.

    Checks each file against its branch schema, then asserts that the two
    branches' full ``sample_index`` sets match exactly (the join keys).
    The comparison is over the UNION of each branch's part files - the
    two branches bucket their shards independently, so individual part
    files are NOT expected to align.

    Returns:
        ``(ctx_rows, target_rows)``.
    """
    n_ctx = sum(validate_parquet(p, "ctx", hidden_size, max_rows_to_check) for p in ctx_paths)
    n_tgt = sum(validate_parquet(p, "targets", hidden_size, max_rows_to_check) for p in target_paths)

    ctx_keys = sorted({i for p in ctx_paths for i in
                       pq.read_table(p, columns=["sample_index"]).to_pandas()["sample_index"].tolist()})
    tgt_keys = sorted({i for p in target_paths for i in
                       pq.read_table(p, columns=["sample_index"]).to_pandas()["sample_index"].tolist()})

    if ctx_keys != tgt_keys:
        only_ctx = sorted(set(ctx_keys) - set(tgt_keys))
        only_tgt = sorted(set(tgt_keys) - set(ctx_keys))
        raise ValueError(
            f"branch sample_index sets differ: {len(only_ctx)} ctx-only rows, "
            f"{len(only_tgt)} targets-only rows"
        )
    return n_ctx, n_tgt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate precomputed R-JEPA parquet branches "
        "(scans part-*.parquet inside each directory)"
    )
    parser.add_argument(
        "--ctx-dir", required=True,
        help="directory holding the ctx branch part files",
    )
    parser.add_argument(
        "--targets-dir", required=True,
        help="directory holding the targets branch part files",
    )
    parser.add_argument(
        "--hidden-size", type=int, required=True,
        help="encoder width E used by the precompute runs",
    )
    parser.add_argument(
        "--max-rows-to-check", type=int, default=512,
        help="rows sampled for blob-shape consistency per file",
    )

    args = parser.parse_args()

    ctx_paths = sorted(str(p) for p in Path(args.ctx_dir).glob("part-*.parquet"))
    tgt_paths = sorted(str(p) for p in Path(args.targets_dir).glob("part-*.parquet"))
    if not ctx_paths or not tgt_paths:
        raise SystemExit(
            f"no part-*.parquet files found under {args.ctx_dir} and {args.targets_dir}"
        )

    print(f"ctx files:     {[Path(p).name for p in ctx_paths]}")
    print(f"targets files: {[Path(p).name for p in tgt_paths]}")

    n_ctx, n_tgt = validate_pair(
        ctx_paths, tgt_paths, args.hidden_size, args.max_rows_to_check,
    )
    print(f"validation successful: {n_ctx:,} ctx rows, {n_tgt:,} target rows")