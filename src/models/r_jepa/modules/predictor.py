"""JEPA predictor modules for autoregressive latent-state forecasting.

Two predictor variants are provided:

* **CrossAttentionPredictor** - reasoning tokens attend to themselves
  causally (self-attention) and cross-attend to frozen context
  representations (the instruction encoding). Best suited when context
  and reasoning operate at different semantic levels.

* **CausalAttentionPredictor** - context, a learnable start token, and
  reasoning tokens are concatenated into one causal sequence. Simpler
  architecture; the model learns to transition from context encoding to
  latent-state reasoning within a single attention stream.

Both support per-sample padding masks so that variable-length batches
can be handled without waste (see :class:`MultiHeadAttention` for mask
composition details).
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn

from src.models.modules.block import SelfAttentionBlock, CrossAttentionBlock, RMSNorm


class RJEPAPredictor(nn.Module, ABC):
    """Shared base for R-JEPA predictors.

    Provides:
    * Learnable start token that seeds the autoregressive chain.
    * Down/up projections between encoder and predictor dimensions.
    * Router that scores each latent state for dynamic halting.
    * Precomputed RoPE frequencies.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int, num_heads: int,
                 max_seq_length: int):
        super(RJEPAPredictor, self).__init__()
        self.start_token = nn.Parameter(
            torch.randn(1, 1, predictor_dim)
        )

        self.context_down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.reasoning_down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

        self.router = nn.Linear(predictor_dim, 1)

        # Precompute RoPE frequencies.
        head_dim = predictor_dim // num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        t = torch.arange(max_seq_length)

        angles = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Causal mask: lower-triangular, sized for the full predictor window.
        mask = (
            torch.tril(torch.ones(max_seq_length, max_seq_length))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("mask", mask, persistent=False)

    @abstractmethod
    def forward(self, x, context=None, **kwargs):
        pass


class CrossAttentionPredictor(RJEPAPredictor):
    """Predictor that cross-attends reasoning tokens to a frozen context.

    Architecture per layer: self-attention (causal) → cross-attention → FFN.

    The reasoning sequence is ``[start_token, step_0, step_1, …, step_{N-1}]``.
    Self-attention is causal within this sequence. Cross-attention reads from
    the context (instruction encoding) at every layer.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int, num_heads: int,
                 d_ff: int, num_layers: int, max_seq_length: int,
                 dropout: float = 0.0):
        super(CrossAttentionPredictor, self).__init__(
            encoder_dim, predictor_dim, num_heads, max_seq_length,
        )
        self.layers = nn.ModuleList([
            CrossAttentionBlock(predictor_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor] = None,
        step_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional per-sample padding masks.

        Args:
            x: ``(B, S_input, encoder_dim)`` reasoning-step representations
                (autoregressive targets without the final step).
            context: ``(B, ctx_len, encoder_dim)`` context (instruction)
                representations.
            context_padding_mask: ``(B, ctx_len)`` bool, **True** = padding
                on the context / cross-attention KV side.
            step_padding_mask: ``(B, S_input)`` bool, **True** = padding
                on the reasoning-step input side.

        Returns:
            ``(predictions, router_logits)`` where
            ``predictions`` is ``(B, S, encoder_dim)`` (S = 1 + S_input;
            includes the start-token output) and
            ``router_logits`` is ``(B, S)`` with raw halting scores.
        """
        B = x.size(0)

        # Project to predictor dimension.
        context = self.context_down_projection(context)   # (B, ctx_len, P)
        x = self.reasoning_down_projection(x)             # (B, S_in, P)

        # Prepend the learnable start token.
        x = torch.cat([self.start_token.expand(B, -1, -1), x], dim=1)
        # → (B, S, P)  where S = 1 + S_in

        # Build reasoning-side (self-attention) padding mask.
        if step_padding_mask is not None:
            start_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            self_mask = torch.cat([start_mask, step_padding_mask], dim=1)
        else:
            self_mask = None

        # Build per-batch RoPE frequencies for the reasoning side.
        S = x.size(1)
        batch_freqs_cis = self.freqs_cis[:S]  # (S, head_dim)

        # Causal mask sliced to the batch's actual reasoning length.
        causal_mask = self.mask[:, :, :S, :S]

        for layer in self.layers:
            x = layer(
                x, context=context,
                mask=causal_mask, freqs_cis=batch_freqs_cis,
                self_attn_padding_mask=self_mask,
                cross_attn_padding_mask=context_padding_mask,
            )
        x = self.norm(x)

        router_logits = self.router(x).squeeze(-1)  # (B, S)
        x = self.up_projection(x)                   # (B, S, E)

        return x, router_logits


class CausalAttentionPredictor(RJEPAPredictor):
    """Predictor that concatenates context + start token + reasoning into a
    single causal sequence.

    Architecture per layer: self-attention (causal) → FFN. No separate
    cross-attention - the model learns to transition from context to
    reasoning within one attention stream.

    The full sequence is
    ``[ctx_0, …, ctx_{C-1}, <start>, step_0, …, step_{N-1}]``.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int, num_heads: int,
                 d_ff: int, num_layers: int, max_seq_length: int,
                 dropout: float = 0.0):
        super(CausalAttentionPredictor, self).__init__(
            encoder_dim, predictor_dim, num_heads, max_seq_length,
        )
        self.layers = nn.ModuleList([
            SelfAttentionBlock(predictor_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor] = None,
        step_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass with optional per-sample padding masks.

        Args:
            x: ``(B, S_input, encoder_dim)`` reasoning-step representations
                (autoregressive targets without the final step).
            context: ``(B, ctx_len, encoder_dim)`` context (instruction)
                representations.
            context_padding_mask: ``(B, ctx_len)`` bool, **True** = padding
                on the context side.
            step_padding_mask: ``(B, S_input)`` bool, **True** = padding
                on the reasoning-step input side.

        Returns:
            ``(predictions, router_logits)`` where
            ``predictions`` is ``(B, S, encoder_dim)`` (S = 1 + S_input;
            includes the start-token output) and
            ``router_logits`` is ``(B, S)`` - only the reasoning-step
            positions (context positions are discarded).
        """
        B = x.size(0)
        ctx_len = context.size(1)

        # Project to predictor dimension.
        context = self.context_down_projection(context)   # (B, ctx_len, P)
        x = self.reasoning_down_projection(x)             # (B, S_in, P)

        # Concatenate: context → start_token → reasoning steps.
        x = torch.cat([
            context,
            self.start_token.expand(B, -1, -1),
            x,
        ], dim=1)  # (B, ctx_len + 1 + S_in, P)

        total_len = x.size(1)
        S = total_len - ctx_len  # 1 + S_in: reasoning-step positions

        # Build total per-sample padding mask.
        cp = (
            context_padding_mask
            if context_padding_mask is not None
            else torch.zeros(B, ctx_len, dtype=torch.bool, device=x.device)
        )
        sp = (
            step_padding_mask
            if step_padding_mask is not None
            else torch.zeros(B, S - 1, dtype=torch.bool, device=x.device)
        )
        start_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        total_padding_mask = torch.cat([cp, start_mask, sp], dim=1)  # (B, total_len)

        # Per-batch RoPE frequencies and causal mask.
        batch_freqs_cis = self.freqs_cis[:total_len]          # (total_len, head_dim)
        causal_mask = self.mask[:, :, :total_len, :total_len]  # (1, 1, total_len, total_len)

        for layer in self.layers:
            x = layer(
                x, mask=causal_mask, freqs_cis=batch_freqs_cis,
                key_padding_mask=total_padding_mask,
            )
        x = self.norm(x)

        # Router operates on all positions; we keep only the reasoning-step
        # scores so the interface matches CrossAttentionPredictor.
        all_router_logits = self.router(x).squeeze(-1)    # (B, total_len)
        router_logits = all_router_logits[:, ctx_len:]     # (B, S)

        x = self.up_projection(x)
        # Return only the reasoning-step positions (excluding context).
        return x[:, ctx_len:, :], router_logits
