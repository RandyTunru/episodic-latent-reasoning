"""Length-bucketed batch sampling for R-JEPA training.

Batch composition is the lever that controls padding waste at training
time: a batch's cost is driven by its longest sample, so batches should
group samples of similar length.  The sort key is variant-specific:

* causal  - the packed stream length ``ctx_len + num_steps`` (the batch
  pads to the max TOTAL, so the total is the true length axis);
* cross   - ``num_steps`` primary (self-attention is quadratic in the
  reasoning stream), ``ctx_len`` secondary.

Samples are assigned to buckets by quantizing the primary key against
geometric bucket edges, then per epoch: each bucket is shuffled in
windows (keeps local length homogeneity while randomizing composition
across epochs), chunked into batches, and the batch list is shuffled.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

# Geometric edges: bucket k covers (edges[k-1], edges[k]].  Doubling
# edges keep within-bucket padding waste below ~2x.
DEFAULT_BUCKET_EDGES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


class BucketedBatchSampler(Sampler):
    """Yield shuffled batches of length-homogeneous samples.

    Args:
        primary: ``(N,)`` integer lengths the batches are bucketed on.
        secondary: Optional ``(N,)`` lengths used to sort *within* each
            bucket (tightens the second padding axis where one exists).
        batch_size: Maximum samples per batch (tail chunks may be smaller
            unless ``drop_last``).
        bucket_edges: Ascending bucket boundaries (default: geometric).
        window: Shuffle window size within a bucket, in samples.  The
            window is shuffled before batching so consecutive epochs see
            different compositions while staying length-homogeneous.
        drop_last: Drop the final partial batch of each bucket.
        seed: Seed for the per-epoch shuffles (None = nondeterministic).
    """

    def __init__(
        self,
        primary: Sequence[int],
        secondary: Optional[Sequence[int]] = None,
        batch_size: int = 16,
        bucket_edges: Optional[Sequence[int]] = None,
        window: int = 1024,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.primary = np.asarray(primary, dtype=np.int64)
        self.secondary = (
            np.asarray(secondary, dtype=np.int64) if secondary is not None else self.primary
        )
        assert self.primary.shape == self.secondary.shape, "primary/secondary length mismatch"
        self.batch_size = batch_size
        self.window = window
        self.drop_last = drop_last
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.bucket_edges = sorted(bucket_edges or DEFAULT_BUCKET_EDGES)
        if self.primary.max() >= self.bucket_edges[-1]:
            self.bucket_edges = self.bucket_edges + [int(self.primary.max()) + 1]

        # Bucket assignment: bucket b = index of first edge > key.
        self.buckets: List[np.ndarray] = []
        for b in range(len(self.bucket_edges) - 1):
            lo, hi = self.bucket_edges[b], self.bucket_edges[b + 1]
            members = np.nonzero((self.primary > lo) & (self.primary <= hi))[0]
            if members.size:
                self.buckets.append(members)

    def __len__(self) -> int:
        n = 0
        for members in self.buckets:
            n += (members.size + self.batch_size - 1) // self.batch_size
        return n

    def __iter__(self):
        batches: List[np.ndarray] = []
        for members in self.buckets:
            # Sort by secondary key, then shuffle within windows so the
            # batches stay length-homogeneous but vary across epochs.
            order = np.argsort(self.secondary[members], kind="stable")
            members = members[order]
            for start in range(0, members.size, self.window):
                window = members[start : start + self.window]
                self.rng.shuffle(window)
                members[start : start + self.window] = window

            chunks = [
                members[i : i + self.batch_size]
                for i in range(0, members.size, self.batch_size)
            ]
            if self.drop_last and chunks and chunks[-1].size < self.batch_size:
                chunks = chunks[:-1]
            batches.extend(chunks)

        # Randomize the order batches are seen in.
        order = self.rng.permutation(len(batches))
        for i in order:
            yield torch.as_tensor(batches[i], dtype=torch.long).tolist()
