"""Universal dataset for reading precomputed R-JEPA representation files.

Any text dataset that has been run through both branches of
``openr1_reasoning_precompute.py`` (``--branch ctx`` and ``--branch
targets``) is readable by this class, regardless of the original text
source - the branch parquet contract in ``src.data.helpers.precomputed``
IS the data format, and the original source never enters this module.
Point the constructor at the two branch directories and you get a
map-style ``torch.utils.data.Dataset``.

Mechanics:

* Construction reads only the tiny ``sample_index`` columns (~1.4 MB for
  the full science split) and builds key -> row-position maps per branch.
* The two branches are joined on ``sample_index``; :func:`validate_pair`
  confirms the key relationship before anything else is read - exact
  1:1 equality for standard data, or the relaxed 1:k form for rewritten
  multi-variant data (see ``n_rewrite`` below).
* Blob payloads are decoded per row on demand via ``torch.frombuffer`` -
  zero-copy (read-only) views over the decoded row bytes.
* Row order is arbitrary and per-epoch shuffling happens in the DataLoader
  as usual; each branch's own bucketing/rank layout is absorbed by the
  position maps, so per-part file contents never matter here.
"""

from pathlib import Path
from typing import Optional, Union

import bisect
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.data.helper.precomputed import validate_pair


def _build_row_group_index(paths):
    """Flat row-group index over the concatenation of *paths* in order.

    Returns ``(entries, starts)`` where ``entries[k] = (ParquetFile,
    row_group_idx, global_start_row, rows_in_group)`` and ``starts[k]``
    is the global position of the first row of entry ``k``.
    """
    entries, starts = [], []
    pos = 0
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg in range(pf.metadata.num_row_groups):
            rows = pf.metadata.row_group(rg).num_rows
            entries.append((pf, rg, pos, rows))
            starts.append(pos)
            pos += rows
    return entries, starts


class PrecomputedReasoningDataset(Dataset):
    """Map-style dataset over precomputed ``ctx`` + ``targets`` branches.

    Args:
        ctx_dir: Directory holding the ``ctx`` branch's ``part-*.parquet``
            files (e.g. ``data/openr1_science_precompute/ctx``).
        targets_dir: Directory holding the ``targets`` branch's
            ``part-*.parquet`` files.
        hidden_size: Encoder width ``E`` used by the precompute run
            (blob shapes are recovered as ``len(blob) // (E * 2)``).
        max_steps: Optional cap on reasoning steps per sample, applied
            AT LOAD (training).  The stored data is untouched; the
            sample's ``num_steps`` clamps while ``num_steps_true`` keeps
            the true count, so the trainer can tell whether the true end
            (the router's halt position) is inside the visible window.
        max_ctx_len: Optional cap on ctx tokens per sample, applied at
            load (training).  Slices the stored embeddings - no
            re-encoding.
        n_rewrite: Rewrite variants per instruction (the k in the
            precompute's composite key ``orig_row_id * k + cot_variant``).
            ``1`` (the default when unset and no sidecar exists) = standard
            data: each row is its own group, the join is exact 1:1.
            ``> 1`` = rewritten multi-variant data: the targets branch
            holds k rows per instruction (composite keys) while the ctx
            branch holds ONE row per instruction (key = orig_row_id), and
            every variant row resolves its ctx through
            ``ctx_pos[key // n_rewrite]`` - the stored ctx is shared k
            ways, never duplicated.  The ctx keys must be the dense
            enumeration ``0..N-1`` (checked at init).  When None, the
            value is read from the targets branch's part-*.json stats
            sidecar (which records the key formula), else defaults to 1.
        validate: Run :func:`validate_pair` before reading (cheap; reads
            schemas and key columns only).

    Each sample returns:
        ctx_embeddings: ``(ctx_len, E)`` bf16 - token-level instruction
            encodings, unpadded.
        step_targets: ``(num_steps, E)`` bf16 - pooled per-step encodings,
            unpadded.
        ctx_len: int - number of instruction tokens.
        num_steps: int - effective reasoning steps, ``min(X, P, T)``
            (storage cap, load cap).
        num_steps_true: int - the true count X, never capped; where the
            episode's halt really is.
    """

    def __init__(
        self,
        ctx_dir: Union[str, Path],
        targets_dir: Union[str, Path],
        hidden_size: int,
        max_steps: Optional[int] = None,
        max_ctx_len: Optional[int] = None,
        n_rewrite: Optional[int] = None,
        validate: bool = True,
    ):
        ctx_paths = sorted(Path(ctx_dir).glob("part-*.parquet"))
        tgt_paths = sorted(Path(targets_dir).glob("part-*.parquet"))
        if not ctx_paths or not tgt_paths:
            raise FileNotFoundError(
                f"no part-*.parquet files found under {ctx_dir} and {targets_dir}"
            )

        # The precompute records the key formula in its per-part stats
        # sidecar; when the caller doesn't pass n_rewrite explicitly,
        # read it from there (single source of truth).  Standard runs
        # predating the sidecar field default to 1.
        if n_rewrite is None:
            n_rewrite = 1
            part_jsons = sorted(Path(targets_dir).glob("part-*.json"))
            if part_jsons:
                try:
                    n_rewrite = int(
                        json.loads(part_jsons[0].read_text()).get("n_rewrite", 1)
                    )
                except (OSError, ValueError):
                    n_rewrite = 1
        if n_rewrite < 1:
            raise ValueError(f"n_rewrite must be >= 1, got {n_rewrite}")
        self.n_rewrite = n_rewrite

        if validate:
            validate_pair(
                [str(p) for p in ctx_paths], [str(p) for p in tgt_paths],
                hidden_size, n_rewrite=n_rewrite,
            )

        # Row-group index over the part files (metadata only).  Per-row
        # fetches decode just the containing 32-row group (~4-10 MB)
        # instead of asking the dataset scanner to locate a single row.
        self._ctx_entries, self._ctx_starts = _build_row_group_index(ctx_paths)
        self._tgt_entries, self._tgt_starts = _build_row_group_index(tgt_paths)
        self._ctx_cache = (None, None)  # (group key, decoded table)
        self._tgt_cache = (None, None)
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.max_ctx_len = max_ctx_len

        # Eager part: key + length columns only (~4 MB for the full
        # split).  The lengths feed the bucketed batch sampler.
        ctx_cols = pq.read_table(ctx_paths, columns=["sample_index", "ctx_len"]).to_pandas()
        tgt_cols = pq.read_table(
            tgt_paths, columns=["sample_index", "num_steps_stored"],
        ).to_pandas()
        ctx_keys = ctx_cols["sample_index"].tolist()
        tgt_keys = tgt_cols["sample_index"].tolist()
        self.ctx_pos = {k: i for i, k in enumerate(ctx_keys)}
        self.tgt_pos = {k: i for i, k in enumerate(tgt_keys)}
        # The dataset's rows are the TARGETS rows.  With n_rewrite > 1 a
        # variant's ctx row is its group's single ctx row (key //
        # n_rewrite); with n_rewrite == 1 the sets are equal (validated
        # above) and this is the standard 1:1 join.
        self.keys = sorted(self.tgt_pos)

        # The ctx branch must be the dense enumeration 0..N-1 for the
        # 1:k join (a targets key k resolves ctx via k // n_rewrite, so
        # ctx keys must BE the group ids).
        if n_rewrite > 1 and sorted(ctx_keys) != list(range(len(ctx_keys))):
            raise ValueError(
                "ctx branch keys must be the dense enumeration 0..N-1 to "
                "serve as orig_row_id for the n_rewrite join; found gaps "
                "or reordering"
            )

        # Length arrays aligned with self.keys order (for the sampler).
        ctx_len_by_key = dict(zip(ctx_keys, ctx_cols["ctx_len"].tolist()))
        # Bucketing key: num_steps_stored records min(X, P) - the exact
        # effective length before the load cap - so the key equals what
        # each sample actually costs at train time, with no blob
        # decoding.  num_steps (X) stays in the column untouched as the
        # router's halt signal.
        stored_by_key = dict(zip(tgt_keys, tgt_cols["num_steps_stored"].tolist()))
        # Drop samples with no stored steps: their step_targets blob is
        # empty, which torch.frombuffer refuses (and a zero-step episode
        # has no halt position for the router loss anyway).  Such rows
        # appear when the generator produced no reasoning steps.
        n_all = len(self.keys)
        self.keys = [k for k in self.keys if stored_by_key[k] > 0]
        if len(self.keys) < n_all:
            print(
                f"PrecomputedReasoningDataset: dropped {n_all - len(self.keys)} "
                f"of {n_all} samples with num_steps_stored == 0"
            )
        ctx_lens = np.array(
            [ctx_len_by_key[k // n_rewrite] for k in self.keys], dtype=np.int64,
        )
        num_steps_arr = np.array(
            [stored_by_key[k] for k in self.keys], dtype=np.int64
        )
        if max_ctx_len is not None:
            ctx_lens = np.minimum(ctx_lens, max_ctx_len)
        if max_steps is not None:
            num_steps_arr = np.minimum(num_steps_arr, max_steps)
        self.ctx_lens = ctx_lens
        self.num_steps_arr = num_steps_arr

    def __len__(self) -> int:
        return len(self.keys)

    def bucket_keys(self, mode: str = "causal"):
        """Length keys for `BucketedBatchSampler`.

        Args:
            mode: ``"causal"`` → packed stream total ``ctx_len +
                num_steps`` (the batch pads to the max total, so the
                total is the true length axis).  ``"cross"`` →
                ``num_steps`` primary with ``ctx_len`` secondary.

        Returns:
            ``(primary, secondary)`` numpy arrays; ``secondary`` is
            ``None`` for ``"causal"``.
        """
        if mode == "causal":
            return self.ctx_lens + self.num_steps_arr, None
        if mode == "cross":
            return self.num_steps_arr, self.ctx_lens
        raise ValueError(f"unknown bucketing mode '{mode}' (expected 'causal' or 'cross')")

    def _fetch(self, cache_attr: str, entries, starts, pos: int) -> dict:
        """Decode the row at global position *pos* via its row group."""
        k = bisect.bisect_right(starts, pos) - 1
        pf, rg, start, _rows = entries[k]
        group_key = (pf.metadata.num_rows, rg, start)
        cached_key, cached = getattr(self, cache_attr)
        if cached_key != group_key:
            cached = pf.read_row_group(rg)
            setattr(self, cache_attr, (group_key, cached))
        return cached.slice(pos - start, 1).to_pydict()

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]

        # Follow the join key into each branch's physical layout.  With
        # n_rewrite > 1 every variant row resolves its ctx through its
        # group's single ctx row (key // n_rewrite); with n_rewrite == 1
        # this is the plain 1:1 join (key // 1 == key).
        ctx_row = self._fetch(
            "_ctx_cache", self._ctx_entries, self._ctx_starts,
            self.ctx_pos[key // self.n_rewrite],
        )
        tgt_row = self._fetch("_tgt_cache", self._tgt_entries, self._tgt_starts, self.tgt_pos[key])

        # The column records the TRUE step count (X) - the router's halt
        # signal.  The stored blob may hold fewer rows when a storage cap
        # was used, so the effective length is clamped to what is actually
        # stored (and to the load-time cap).
        num_steps_true = int(tgt_row["num_steps"][0])
        tgt = torch.frombuffer(
            tgt_row["step_targets"][0], dtype=torch.bfloat16,
        ).reshape(-1, self.hidden_size)  # min(X, P) rows
        if self.max_steps is not None:
            tgt = tgt[: self.max_steps]
        num_steps = tgt.size(0)  # effective min(X, P, T)

        ctx = torch.frombuffer(
            ctx_row["ctx_embeddings"][0], dtype=torch.bfloat16,
        ).reshape(-1, self.hidden_size)  # (ctx_len, E)
        if self.max_ctx_len is not None:
            ctx = ctx[: self.max_ctx_len]

        return {
            "ctx_embeddings": ctx,   # read-only views; collate copies into padded tensors
            "step_targets": tgt,
            "ctx_len": min(int(ctx_row["ctx_len"][0]), self.max_ctx_len or 2**30),
            "num_steps": num_steps,            # effective min(X, P, T)
            "num_steps_true": num_steps_true,  # X - where the true halt is
        }
