# episodic-latent-reasoning

Compressing verbose Chain-of-Thought reasoning into compact, continuous latent
representations for efficient LLM inference.

## The Problem

Current autoregressive Chain-of-Thought (CoT) prompting allows LLMs to solve
complex problems by explicitly generating intermediate thinking blocks
textually. However, this introduces two bottlenecks:

1. **Quadratic Scaling (O(N²))** - textual thinking blocks routinely consume
   thousands of tokens, scaling compute costs quadratically.
2. **Syntax Coupling** - standard CoT couples abstract logical reasoning with
   linguistic syntax; minor grammatical slips can derail the entire reasoning
   trajectory.

## The Approach

This project decouples logical planning from syntactic token generation using a
**Joint Embedding Predictive Architecture (JEPA)**:

1. A **frozen encoder** extracts semantic context from the instruction.
2. An **autoregressive predictor** iteratively forecasts sequential latent
   reasoning states starting from a learnable `<start>` token.
3. A **lightweight router** evaluates each state and triggers dynamic halting
   once the logic converges.

The predicted latent states can be injected directly into a primary LLM's
context window as continuous embeddings (soft-prompting), allowing the LLM to
attend to dense reasoning representations alongside standard textual tokens.

## Architecture

```
Instruction → [Frozen Encoder] → context_repr
                                      │
     ┌────────────────────────────────┘
     ▼
[Predictor]
  ┌─────────┐   ┌─────────┐        ┌─────────┐
  │ <start> │──▶│ step_0  │──▶ … ──▶│ step_n  │  (autoregressive)
  └─────────┘   └─────────┘        └────┬────┘
       ▲                                │
       └──── context cross-attn ────────┘
     ┌──────────────────────────────────┘
     ▼
[Router] → halt if logit < 0
```

See [src/models/r_jepa/README.md](src/models/r_jepa/README.md) for detailed
architecture documentation, including predictor variants (cross-attention vs.
causal), variable-length batching strategies (padding and FlashAttention varlen
packing), and usage examples.

## Project Structure

```
episodic-latent-reasoning/
├── archive/                         # Archived / scrapped implementations
│   └── text_jepa/                   #   Word-bounded span masking (I-JEPA for text)
│       └── README.md                #   Explanation of what was attempted and why it was dropped
├── src/
│   ├── data/                        # Data pipeline
│   │   ├── r_jepa_dataset.py        #   ReasoningDataset: tokenizes instructions + steps
│   │   └── collate.py               #   pad_collate: 2D padding (steps × tokens)
│   ├── models/
│   │   ├── modules/                 # Shared building blocks
│   │   │   ├── attention.py         #   MultiHeadAttention, CrossAttention (+ RoPE, padding masks)
│   │   │   ├── block.py             #   SelfAttentionBlock, CrossAttentionBlock, RMSNorm
│   │   │   ├── encoder.py           #   Custom Transformer encoder (token embeds + RoPE + self-attn)
│   │   │   └── ffn.py               #   PositionwiseFeedForward (SwiGLU variant)
│   │   └── r_jepa/                  # R-JEPA model
│   │       ├── r_jepa.py            #   Main RJEPA model (forward + inference)
│   │       ├── modules/
│   │       │   └── predictor.py     #   CrossAttentionPredictor, CausalAttentionPredictor
│   │       └── README.md            #   Detailed architecture docs
│   ├── trainer/
│   │   └── r_jepa/
│   │       └── r_jepa_trainer.py    # Training loop (AMP, grad accum, cosine LR, checkpointing)
│   └── utils/
│       ├── metrics.py               #   ThroughputMonitor, param counting
│       └── performance.py           #   F1 score, accuracy
└── project.txt                      # Project premise and motivation
```

## References

- [Hao et al. - Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [Geiping et al. - Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [Su et al. - Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning](https://arxiv.org/abs/2502.03275)
- [Tan et al. - Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains](https://arxiv.org/abs/2505.16552)
- [Assran et al. - V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- [Bae et al. - Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524v1)
- [Bui et al. - Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems](https://arxiv.org/abs/2507.20491v1)
- [Huang et al. - LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
- [Liu et al. - JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation](https://arxiv.org/abs/2512.19171)
- [Chen et al. - ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning of Large Language Model](https://arxiv.org/abs/2601.22730)
- [Amos et al - Latent Reasoning with Supervised Thinking States](https://arxiv.org/abs/2602.08332)