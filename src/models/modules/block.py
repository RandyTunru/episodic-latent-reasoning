from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules.attention import MultiHeadAttention, CrossAttention
from src.models.modules.ffn import PositionwiseFeedForward


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(SelfAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Self-attention block with optional causal and padding masks.

        Args:
            x: ``(B, seq_len, d_model)`` input.
            mask: ``(1, 1, max_len, max_len)`` causal mask (True = keep).
            freqs_cis: ``(seq_len, head_dim)`` RoPE frequencies.
            key_padding_mask: ``(B, seq_len)`` bool where **True** = padding.

        Returns:
            ``(B, seq_len, d_model)`` output.
        """
        attn_output = self.attention(
            self.norm1(x), mask=mask, freqs_cis=freqs_cis,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        cross_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Cross-attention block: self-attention on *x*, then cross-attention to *context*.

        Args:
            x: ``(B, q_len, d_model)`` query-side input (e.g. reasoning steps).
            context: ``(B, kv_len, d_model)`` key/value-side input (e.g. instruction).
            mask: ``(1, 1, max_len, max_len)`` causal mask for self-attention.
            freqs_cis: ``(q_len, head_dim)`` RoPE frequencies for self-attention.
            self_attn_padding_mask: ``(B, q_len)`` bool where **True** = padding
                on the query (self-attention) side.
            cross_attn_padding_mask: ``(B, kv_len)`` bool where **True** = padding
                on the context (cross-attention) side.

        Returns:
            ``(B, q_len, d_model)`` output.
        """
        self_attn_output = self.self_attention(
            self.norm1(x), mask=mask, freqs_cis=freqs_cis,
            key_padding_mask=self_attn_padding_mask,
        )
        x = x + self.dropout(self_attn_output)

        cross_attn_output = self.cross_attention(
            self.norm2(x), context,
            key_padding_mask=cross_attn_padding_mask,
        )
        x = x + self.dropout(cross_attn_output)

        ffn_output = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_output)

        return x
