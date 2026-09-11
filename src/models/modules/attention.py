from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module, ABC):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def _apply_rope(self, x, freqs_cis):
        """Apply rotary position embeddings to query or key tensors.

        Args:
            x: (batch_size, num_heads, seq_len, d_k)
            freqs_cis: either (seq_len, d_k/2) - one shared rotation table
                for the whole batch (uniform positions), or
                (batch_size, seq_len, d_k/2) - per-sample per-position
                rotations (packed sequences where each sample's real
                positions start at 0).

        Returns:
            Tensor of same shape as x with RoPE applied.
        """
        assert x.size(-1) == self.d_k, "Last dimension of x must match d_k"
        # freqs_cis has one complex frequency per pair of head dimensions.
        assert freqs_cis.size(-1) == self.d_k // 2, (
            f"freqs_cis last dim ({freqs_cis.size(-1)}) must equal d_k // 2 ({self.d_k // 2})"
        )
        assert freqs_cis.dim() in (2, 3), "freqs_cis must be 2D (shared) or 3D (per-sample)"
        seq_len = x.size(2)
        assert freqs_cis.size(-2) >= seq_len, "freqs_cis must have enough length for the given seq_len"

        if freqs_cis.dim() == 3:
            # (B, seq_len, d_k/2) -> (B, 1, seq_len, d_k/2); per-sample positions.
            freqs_cis = freqs_cis[:, :seq_len].unsqueeze(1)
        else:
            freqs_cis = freqs_cis[:seq_len]  # (seq_len, d_k/2); shared across batch & heads
        freqs_cis = freqs_cis.to(x.device)

        x_reshaped = x.float().view(*x.shape[:-1], self.d_k // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        x_rotated = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)

    @staticmethod
    def _build_padded_mask(
        causal_mask: Optional[torch.Tensor],
        key_padding_mask: torch.Tensor,
        q_len: int,
        k_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compose a causal mask with a per-sample key-padding mask into a single
        float mask consumable by :func:`F.scaled_dot_product_attention`.

        Args:
            causal_mask: ``(1, 1, max_q, max_k)`` bool lower-triangular mask, or
                ``None`` when causality is not required.
            key_padding_mask: ``(B, k_len)`` bool where **True** means the key
                position is padding and should be masked.
            q_len: actual query length (≤ causal_mask size).
            k_len: actual key length (≤ causal_mask size).
            device: target device.

        Returns:
            ``(B, 1, q_len, k_len)`` float mask: ``0.0`` = attend,
            ``-inf`` = masked.
        """
        if causal_mask is not None:
            c_mask = causal_mask[:, :, :q_len, :k_len]
            combined = torch.where(c_mask, 0.0, float("-inf")).to(device)
        else:
            combined = torch.zeros(1, 1, q_len, k_len, device=device)

        pad_additive = torch.where(
            key_padding_mask[:, None, None, :k_len],
            float("-inf"),
            0.0,
        )  # (B, 1, 1, k_len)

        return combined + pad_additive  # (B, 1, q_len, k_len)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Abstract method for forward pass of the attention mechanism.
        Subclasses must implement this method.
        """
        pass


class MultiHeadAttention(Attention):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__(d_model, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Self-attention with optional causal mask and per-sample padding mask.

        Args:
            x: ``(B, seq_len, d_model)`` input.
            mask: ``(1, 1, max_len, max_len)`` bool causal mask (True = keep).
            freqs_cis: ``(seq_len, head_dim)`` RoPE frequencies.
            key_padding_mask: ``(B, seq_len)`` bool where **True** = padding
                (should be masked).

        Returns:
            ``(B, seq_len, d_model)`` attention output.
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if freqs_cis is not None:
            q = self._apply_rope(q, freqs_cis)
            k = self._apply_rope(k, freqs_cis)

        # Compose causal + padding masks into a unified float mask for SDPA.
        if mask is not None or key_padding_mask is not None:
            kp = key_padding_mask if key_padding_mask is not None else torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=x.device
            )
            attn_mask = self._build_padded_mask(mask, kp, seq_len, seq_len, x.device)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=attn_mask
        )

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)

        return output


class CrossAttention(Attention):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__(d_model, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Cross-attention where Q comes from *x* and K/V come from *context*.

        Args:
            x: ``(B, q_len, d_model)`` query input.
            context: ``(B, kv_len, d_model)`` key/value input.
            key_padding_mask: ``(B, kv_len)`` bool where **True** = padding
                (masked) on the context side.

        Returns:
            ``(B, q_len, d_model)`` attention output.
        """
        batch_size, q_len, _ = x.shape
        kv_len = context.size(1)

        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Build per-sample padding mask for the KV (context) side if needed.
        if key_padding_mask is not None:
            attn_mask = self._build_padded_mask(
                None, key_padding_mask, q_len, kv_len, x.device
            )
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, attn_mask=attn_mask
        )

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)

        return output
