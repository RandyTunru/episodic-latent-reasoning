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

    for i, sample in enumerate(batch):
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
    }
