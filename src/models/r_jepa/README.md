# R-JEPA (Reasoning Joint Embedding Predictive Architecture)

Compresses verbose Chain-of-Thought reasoning into compact, continuous latent
representations. A frozen pretrained encoder extracts semantic context from the
instruction; an autoregressive predictor iteratively forecasts sequential latent
reasoning states starting from a learnable `<start>` token; a jointly trained
lightweight router evaluates each state and triggers dynamic halting once the
logic converges.

## Motivation

Current autoregressive Chain-of-Thought suffers from two bottlenecks:

1. **Quadratic scaling** - verbatim textual thinking blocks consume thousands of
   tokens, scaling compute costs O(N²).
2. **Syntax coupling** - logical reasoning is coupled with linguistic syntax;
   minor grammatical slips can derail the entire trajectory.

R-JEPA decouples logical planning from syntactic token generation by
compressing reasoning steps into a continuous latent space, achieving
sub-quadratic inference without sacrificing logical depth.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen encoder (no EMA target) | Follows V-JEPA2 action-conditioned phase; both context and target encoders are identical frozen copies |
| Padding over packing (v1) | Simpler to implement and debug; upgrade to FlashAttention varlen when padding waste exceeds 30% |
| Router trained via position pseudo-labels | Intermediate steps = 1 (continue), last step = 0 (halt); no external labels needed |
| Two predictor variants | Cross-attention separates semantics; causal is simpler and packs cleanly |
| Pre-computable encoder representations | Frozen encoder means all context/step embeddings can be computed offline, making training 5-10× faster |

## Architecture

```
┌─────────────┐     ┌──────────────────────────────┐
│  Instruction │────▶│  Frozen Encoder               │
│  (text)      │     │  (e.g., custom Transformer)   │
└─────────────┘     └────────────┬─────────────────┘
                                 │ context_repr
                                 │ (B, ctx_len, E)
                                 ▼
┌─────────────────────────────────────────────────┐
│              Predictor (trainable)               │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ <start>  │──▶│  step_0  │──▶│  step_1  │──▶ … │
│  └──────────┘   └──────────┘   └────┬─────┘     │
│       ▲                             │            │
│       └─── context cross-attn ──────┘            │
│                                                  │
│  Router: score each state → halt if < 0          │
└─────────────────────────────────────────────────┘
```

### Components

| Component | Description | Trainable |
|-----------|-------------|-----------|
| `Encoder` | Custom Transformer with token embeddings, RoPE, RMSNorm, SwiGLU FFN | Frozen |
| `Predictor` | Autoregressive latent-state forecaster (cross-attention or causal) | ✓ |
| `Router` | Linear layer scoring each predicted state for halting | ✓ |
| `start_token` | Learnable parameter seeding the autoregressive chain | ✓ |

### Predictor Variants

**CrossAttentionPredictor** - Reasoning tokens attend to themselves causally
(self-attention) and cross-attend to the frozen context representation at every
layer. Best suited when context and reasoning operate at different semantic
levels.

```
Layer:  Self-Attn(causal) → Cross-Attn(to context) → SwiGLU FFN
Input:  [<start>, step_0, step_1, …, step_{N-1}]
```

**CausalAttentionPredictor** - Context, start token, and reasoning tokens are
concatenated into a single causal sequence. Simpler architecture; the model
learns to transition from context to reasoning within one attention stream.

```
Layer:  Self-Attn(causal) → SwiGLU FFN
Input:  [ctx_0, …, ctx_{C-1}, <start>, step_0, …, step_{N-1}]
```

## Variable-Length Batching

Real-world reasoning data is inherently variable-length - different
instructions have different token counts, and different problems require
different numbers of reasoning steps. Two strategies are supported:

### Strategy 1: Padding (current default)

Sequences are padded to batch-level maxima. A `key_padding_mask` (bool tensor,
`True` = padding) is passed through every attention layer. The attention module
composes the causal mask with the padding mask into a unified float mask
(`0.0` = attend, `-inf` = masked) consumable by `F.scaled_dot_product_attention`.

**Masks in the pipeline:**

| Mask | Source | Shape | Purpose |
|------|--------|-------|---------|
| `ctx_attention_mask` | Collate | `(B, max_ctx_len)` | `True` = valid instruction token |
| `step_attention_mask` | Collate | `(B, max_steps, max_step_len)` | `True` = valid reasoning token |
| `context_padding_mask` | Derived (`~ctx_attention_mask`) | `(B, max_ctx_len)` | `True` = pad context position |
| `step_padding_mask` | Derived (zero-token steps) | `(B, max_steps)` | `True` = pad step |
| `step_valid_mask` | Returned by model | `(B, max_steps)` | Loss masking |

**Loss masking:**
- **MSE**: Only computed at positions where `step_valid_mask` is `True`.
- **Router BCE**: Pseudo-labels - all valid intermediate steps = 1 (continue),
  last valid step = 0 (halt), padding positions ignored via `weight` mask.

### Strategy 2: Packing with FlashAttention Varlen (future upgrade)

For datasets with high length variance, padding waste can exceed 30-50%.
Packing eliminates this by concatenating all valid tokens across the batch into
a single 2D tensor of shape `(total_valid_tokens, embed_dim)`, with
`cu_seqlens` tracking sample boundaries.

```python
# FlashAttention varlen interface (conceptual)
from flash_attn import flash_attn_varlen_func

flash_attn_varlen_func(
    q, k, v,                       # (total_tokens, nheads, head_dim)
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len,
    max_seqlen_k=max_len,
    causal=True,                   # block-diagonal within each sample
)
```

**For the CausalAttentionPredictor** this maps cleanly - each sample is one
block in the packed sequence, and the causal mask is block-diagonal.

**For the CrossAttentionPredictor** both the Q-side (reasoning tokens) and
KV-side (context tokens) must be packed with separate `cu_seqlens`, which
FlashAttention varlen supports natively.

**Migration path:**
1. Pre-compute all encoder representations offline (encoder is frozen).
2. Pack predictor inputs into flat tensors with `cu_seqlens`.
3. Replace `F.scaled_dot_product_attention` with `flash_attn_varlen_func`.
4. Adjust loss computation to operate on the packed layout.

## Data Pipeline

### Dataset (`src/data/r_jepa_dataset.py`)

`ReasoningDataset` accepts a list of ``{instruction, response}`` dicts and a
HuggingFace tokenizer.  Chain-of-thought text is extracted from the response by
stripping configurable think tokens (default: ``<think>`` and ``</think>``), then
split into reasoning steps using a configurable separator.

```python
dataset = ReasoningDataset(
    data=raw_samples,
    tokenizer=tokenizer,
    instruction_key="instruction",
    response_key="response",
    open_think_token="think",
    close_think_token="/think",
    max_ctx_length=512,
    max_step_length=256,
)
```

### Collate (`src/data/collate.py`)

`pad_collate` pads across two dimensions: the number of reasoning steps and
the token length within each step. Returns the four tensors consumed by
`RJEPA.forward()`.

## Training

### Loss Function

```
L = L_MSE + α · L_BCE

L_MSE  = mean over valid steps of ||pred_i - target_i||²
L_BCE  = BCE(router_logit_i, label_i), masked to valid steps
         where label_i = 1 for all valid steps except the last (= 0)
```

### Trainer (`src/trainer/r_jepa/r_jepa_trainer.py`)

Follows the same pattern as the I-JEPA trainers:
- Infinite dataloader iterator
- AMP autocast (bfloat16 on CUDA)
- Gradient accumulation
- Gradient clipping on `trainable_parameters()`
- Cosine LR schedule with linear warmup
- Periodic checkpointing

### Example Usage

```python
from torch.utils.data import DataLoader
from src.models.r_jepa import RJEPA
from src.data.r_jepa_dataset import ReasoningDataset
from src.data.collate import pad_collate
from src.trainer.r_jepa.r_jepa_trainer import Trainer

# Build R-JEPA model (encoder is constructed internally from encoder_kwargs).
model = RJEPA(
    encoder_kwargs=dict(
        vocab_size=32000, d_model=1536, num_heads=16, d_ff=6144,
        num_layers=8, max_seq_length=8192,
    ),
    predictor_kwargs=dict(
        encoder_dim=1536, predictor_dim=1024, num_heads=16,
        d_ff=4096, num_layers=4, max_seq_length=32, dropout=0.1,
    ),
    is_cross_attention=True,
)

# Data pipeline - see ReasoningDataset for think-token extraction.
dataset = ReasoningDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate,
                    shuffle=True, pin_memory=True)

# Train.
trainer = Trainer(
    model=model,
    dataloader=loader,
    optimizer=torch.optim.AdamW(model.trainable_parameters(), lr=1e-4),
    config={
        "learning_rate": 1e-4,
        "max_steps": 100_000,
        "warmup_steps": 2000,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "router_alpha": 0.1,
        "encoder_max_seq_length": 8192,
        "checkpoint_dir": "./checkpoints",
        "log_every": 50,
        "save_every": 5000,
    },
    device=torch.device("cuda"),
)
trainer.train()
```

