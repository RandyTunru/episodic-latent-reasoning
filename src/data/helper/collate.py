from typing import List, Dict, Any

import torch


def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset samples into a padded batch.

    Padding happens along two axes for reasoning steps:
    1. **Steps axis** - pad to the maximum number of reasoning steps in the
       batch.  Empty (all-padding) steps are appended for shorter samples.
    2. **Token axis** - within each step, pad to the maximum token length
       across *all* steps in the batch.

    The instruction (context) is padded to the maximum instruction length in
    the batch. 
    
    Works by pre-allocating a tensor of 0s up to the maximum lengths across both axes,
    and then filling in the values for each sample.

    Returns a dict with keys:
        ctx_input_ids:      ``(B, max_ctx_len)``
        ctx_attention_mask: ``(B, max_ctx_len)`` bool
        step_input_ids:     ``(B, max_steps, max_step_len)``
        step_attention_mask: ``(B, max_steps, max_step_len)`` bool
    """
    # --- batch-level maxima ---
    max_ctx_len = max(s["ctx_ids"].size(0) for s in batch)
    max_steps = max(s["num_steps"] for s in batch)
    max_step_len = max(
        ids.size(0)
        for s in batch
        for ids in s["step_ids"]
    ) if any(s["num_steps"] > 0 for s in batch) else 1

    B = len(batch)

    # Pre-allocate.
    ctx_ids = torch.zeros(B, max_ctx_len, dtype=torch.long)
    ctx_mask = torch.zeros(B, max_ctx_len, dtype=torch.bool)
    step_ids = torch.zeros(B, max_steps, max_step_len, dtype=torch.long)
    step_mask = torch.zeros(B, max_steps, max_step_len, dtype=torch.bool)
    num_steps = torch.zeros(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        num_steps[i] = sample["num_steps"]
        # Instruction (context).
        c_len = sample["ctx_ids"].size(0)
        ctx_ids[i, :c_len] = sample["ctx_ids"]
        ctx_mask[i, :c_len] = sample["ctx_mask"]

        # Reasoning steps - 2D padding.
        for j, (ids, mask) in enumerate(zip(sample["step_ids"], sample["step_masks"])):
            s_len = ids.size(0)
            step_ids[i, j, :s_len] = ids
            step_mask[i, j, :s_len] = mask
        # Remaining step slots (j >= num_steps) stay all-zeros → all-padding.

    return {
        "ctx_input_ids": ctx_ids,
        "ctx_attention_mask": ctx_mask,
        "step_input_ids": step_ids,
        "step_attention_mask": step_mask,
        "num_steps": num_steps,
    }


def pad_collate_precomputed_cross(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate precomputed R-JEPA samples into a padded batch.

    The token axis is gone (pooling collapsed it at precompute time), so
    padding happens along two independent axes to their batch maxima:
    1. **Ctx axis** - token-level instruction encodings.
    2. **Steps axis** - pooled per-step targets.

    The ctx mask is all-True up to each sample's ``ctx_len`` (rows are
    stored unpadded) and ``num_steps`` delimits the real step slots.

    Returns a dict with keys:
        ctx_embeddings:     ``(B, max_ctx_len, E)`` bf16
        ctx_attention_mask: ``(B, max_ctx_len)`` bool
        step_targets:       ``(B, max_steps, E)`` bf16
        num_steps:          ``(B,)`` int64
    """
    B = len(batch)
    E = batch[0]["ctx_embeddings"].size(-1)
    max_ctx_len = max(s["ctx_embeddings"].size(0) for s in batch)
    max_steps = max(s["num_steps"] for s in batch)

    # Pre-allocate.
    ctx = torch.zeros(B, max_ctx_len, E, dtype=torch.bfloat16)
    ctx_mask = torch.zeros(B, max_ctx_len, dtype=torch.bool)
    steps = torch.zeros(B, max_steps, E, dtype=torch.bfloat16)
    num_steps = torch.zeros(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        c_len = sample["ctx_embeddings"].size(0)
        n = sample["num_steps"]
        ctx[i, :c_len] = sample["ctx_embeddings"]
        ctx_mask[i, :c_len] = True
        steps[i, :n] = sample["step_targets"]
        num_steps[i] = n

    return {
        "ctx_embeddings": ctx,
        "ctx_attention_mask": ctx_mask,
        "step_targets": steps,
        "num_steps": num_steps,
    }


def pad_collate_precomputed_causal(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pack precomputed samples into the causal predictor's single stream.

    Each sample's real content is packed as ``[ctx | steps_0..n-2]`` with
    padding only at the end - the predictor inserts its learnable
    ``<start>`` token between ctx and steps.  The whole batch is padded
    to the max TOTAL length ``max_i(ctx_len_i + n_i - 1)``, so there is
    no per-axis bounding-box waste (a batch's cost is driven purely by
    its largest total, which is what the causal sampler buckets on).

    The last step target is never an input (the predictor forecasts it
    autoregressively), which is why the packed stream holds n-1 steps.

    Returns a dict with keys:
        packed_input:     ``(B, T)``-length stream, ``(B, T, E)`` bf16
        packed_valid_mask: ``(B, T)`` bool (True = real content)
        ctx_len:          ``(B,)`` int64 - ctx/step boundary per sample
        step_targets:     ``(B, max_steps, E)`` bf16 - loss targets
        num_steps:        ``(B,)`` int64
    """
    B = len(batch)
    E = batch[0]["ctx_embeddings"].size(-1)

    # Per-sample packed rows: [ctx | steps 0..n-2].
    rows = []
    for s in batch:
        n = s["num_steps"]
        steps_in = s["step_targets"][: max(n - 1, 0)]
        rows.append(torch.cat([s["ctx_embeddings"], steps_in], dim=0))

    T = max(r.size(0) for r in rows)
    packed = torch.zeros(B, T, E, dtype=torch.bfloat16)
    valid = torch.zeros(B, T, dtype=torch.bool)
    ctx_len = torch.zeros(B, dtype=torch.long)
    num_steps = torch.zeros(B, dtype=torch.long)

    for i, (r, s) in enumerate(zip(rows, batch)):
        packed[i, : r.size(0)] = r
        valid[i, : r.size(0)] = True
        ctx_len[i] = s["ctx_embeddings"].size(0)
        num_steps[i] = s["num_steps"]

    # Targets for the loss side, padded along the steps axis only.
    S = max(max(s["num_steps"] for s in batch), 1)
    targets = torch.zeros(B, S, E, dtype=torch.bfloat16)
    for i, s in enumerate(batch):
        targets[i, : s["num_steps"]] = s["step_targets"]

    return {
        "packed_input": packed,
        "packed_valid_mask": valid,
        "ctx_len": ctx_len,
        "step_targets": targets,
        "num_steps": num_steps,
    }
