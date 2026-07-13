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

    def forward(self, x, mask= None, freqs_cis=None):
        attn_output = self.attention(self.norm1(x), mask=mask, freqs_cis=freqs_cis)
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

    def forward(self, x, context, mask=None, freqs_cis=None):
        self_attn_output = self.self_attention(self.norm1(x), mask=mask, freqs_cis=freqs_cis)
        x = x + self.dropout(self_attn_output)

        cross_attn_output = self.cross_attention(self.norm2(x), context)
        x = x + self.dropout(cross_attn_output)

        ffn_output = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_output)

        return x