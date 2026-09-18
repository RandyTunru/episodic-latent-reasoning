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

    def forward_packed(
        self,
        packed: torch.Tensor,
        padding_mask: torch.Tensor,
        ctx_len: torch.Tensor,
        S_out: int,
    ):
        """Forward over per-sample packed streams (precomputed training path).

        The collate packs each sample as ``[ctx | steps_0..n-2]`` with
        trailing padding only; this method inserts the learnable start
        token at each sample's ``ctx_len`` boundary and runs the causal
        stack with per-sample packed RoPE positions - so training
        positions match the unpadded, batch-size-1 inference regime.

        Args:
            packed: ``(B, T, encoder_dim)`` - ctx and step inputs packed
                per sample, trailing padding.
            padding_mask: ``(B, T)`` bool, True = padding.
            ctx_len: ``(B,)`` int - boundary between ctx and steps.
            S_out: number of reasoning positions to gather (the batch's
                padded step count).

        Returns:
            ``(predictions, router_logits)`` where predictions is
            ``(B, S_out, encoder_dim)`` (position j = forecast of step j;
            j = 0 is the start-token output) and router_logits is
            ``(B, S_out)``.  Slots with ``j >= num_steps`` hold garbage -
            the caller masks them with the step validity mask.
        """
        B, T, _ = packed.shape
        device = packed.device

        # Project the two input spaces through their own maps, then
        # select by position: ctx part -> context projection, step part
        # -> reasoning projection.  (Pad slots use the reasoning map;
        # their values are masked downstream.)
        pos = torch.arange(T, device=device).unsqueeze(0)          # (1, T)
        is_ctx = pos < ctx_len.unsqueeze(1)                         # (B, T)

        x_ctx = self.context_down_projection(packed)                # (B, T, P)
        x_stp = self.reasoning_down_projection(packed)              # (B, T, P)
        x = torch.where(is_ctx.unsqueeze(-1), x_ctx, x_stp)

        # Insert the learnable start token at each sample's ctx/step
        # boundary: positions > ctx_len shift right by one.
        T1 = T + 1
        pos1 = torch.arange(T1, device=device).unsqueeze(0)         # (1, T1)
        src_pos = torch.where(
            pos1 > ctx_len.unsqueeze(1), pos1 - 1, pos1.clamp(max=T - 1),
        )                                                            # (B, T1)
        x = x.gather(1, src_pos.unsqueeze(-1).expand(B, T1, x.size(-1)))
        start_mask = pos1 == ctx_len.unsqueeze(1)                    # (B, T1)
        x = torch.where(
            start_mask.unsqueeze(-1), self.start_token.expand(B, T1, -1), x,
        )

        # Masks: real content is valid; the start token is always valid.
        valid = ~padding_mask                                          # (B, T)
        valid = valid.gather(1, src_pos)                               # (B, T1)
        valid = valid | start_mask
        total_padding_mask = ~valid                                    # (B, T1)

        # Per-batch uniform RoPE frequencies and causal mask for the full predictor window.
        batch_freqs_cis = self.freqs_cis[:T1]                          # (T1, d_k/2)
        causal_mask = self.mask[:, :, :T1, :T1]                         

        for layer in self.layers:
            x = layer(
                x, mask=causal_mask, freqs_cis=batch_freqs_cis,
                key_padding_mask=total_padding_mask,
            )
        x = self.norm(x)

        router_all = self.router(x).squeeze(-1)                        # (B, T1)
        x = self.up_projection(x)                                      # (B, T1, E)

        # Gather reasoning positions: output j of sample i sits at packed
        # position ctx_len[i] + j - j = 0 is the start-token output
        # (the forecast of step 0), so it reads position ctx_len[i],
        # which the start token occupies.
        out_pos = ctx_len.unsqueeze(1) + torch.arange(S_out, device=device).unsqueeze(0)
        out_pos = out_pos.clamp(max=T1 - 1)                            # (B, S_out)
        predictions = x.gather(1, out_pos.unsqueeze(-1).expand(B, S_out, x.size(-1)))
        router_logits = router_all.gather(1, out_pos)

        return predictions, router_logits # Only the reasoning-step positions are returned; context is discarded.
