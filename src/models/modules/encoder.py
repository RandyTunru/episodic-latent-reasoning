"""Transformer encoder - a showcase implementation, not a required component.

This is a decoder-style embedding Transformer with RoPE, RMSNorm, and SwiGLU
FFN.  It uses **bidirectional** self-attention (no causal mask), making it
similar to a BERT encoder rather than a GPT decoder.

This file exists for the following reasons:

1. **Audit tool** - every tensor operation is explicit and auditable.
   You can trace from token IDs → embeddings → attention → output without
   wading through HuggingFace's ``modeling_*.py`` files.

2. **Prototyping** - you can verify the full R-JEPA pipeline (dataset →
   forward → loss → backward) without downloading gigabytes of pretrained
   weights.  The encoder runs with random weights and the predictor still
   trains - the latent states won't be *meaningful*, but the tensor math
   is fully exercisable.

3. **Research** - if you think this architecture (SwiGLU +
   RoPE + RMSNorm + bidirectional attention) has advantages over off-the-
   shelf encoders for reasoning tasks, this is the place to pretrain it
   (e.g. via MLM) before freezing it for R-JEPA. Simply make a model that 
   Extends this with a linear head and train it.

For actual experiments of the RJEPA, I decided to swap in a pretrained HuggingFace model via
``RJEPA(encoder=my_pretrained_model, ...)``.  See
``src/models/r_jepa/README.md`` §Encoder for the full swap guide.
"""

from typing import Optional
import torch
from torch import nn

from src.models.modules.block import SelfAttentionBlock, RMSNorm


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_length: int,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Precompute RoPE frequencies up to max_seq_length.
        head_dim = d_model // num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        t = torch.arange(max_seq_length)

        angles = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Encode token sequences into contextualised representations.

        Args:
            x: ``(B, seq_len)`` token IDs.
            attention_mask: ``(B, seq_len)`` bool where **True** means a valid
                token and **False** means padding. If ``None``, all positions
                are treated as valid.

        Returns:
            ``(B, seq_len, d_model)`` encoded representations.
        """
        x = self.token_embedding(x)

        seq_len = x.size(1)

        # RoPE frequencies are position-based (identical across batch members).
        # _apply_rope handles broadcasting from (seq_len, head_dim) to
        # (B, num_heads, seq_len, head_dim) internally.
        batch_freqs_cis = self.freqs_cis[:seq_len]  # (seq_len, head_dim)

        # Convert attention_mask (True = valid) to key_padding_mask (True = masked/padding).
        # If no mask is provided, all positions are valid.
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # True = padding
        else:
            key_padding_mask = torch.zeros(
                x.size(0), seq_len, dtype=torch.bool, device=x.device
            )

        for layer in self.layers:
            x = layer(
                x, mask=None, freqs_cis=batch_freqs_cis,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(x)
        return x
