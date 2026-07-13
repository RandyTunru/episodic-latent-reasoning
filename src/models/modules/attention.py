from abc import ABC, abstractmethod
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

    def _apply_rope(self, x, freqs_cis, starts_from:int = 0):
        # x: (batch_size, num_heads, seq_len, d_k)
        # freqs_cis: (seq_len, d_k)
        assert x.size(-1) == self.d_k, "Last dimension of x must match d_k"
        assert freqs_cis.size(-1) == self.d_k, "Last dimension of freqs_cis must match d_k"
        assert freqs_cis.size(0) >= starts_from + x.size(2), "freqs_cis must have enough length for the given starts_from and seq_len"
        assert freqs_cis.dim() == 2, "freqs_cis must be a 2D tensor"
        seq_len = x.size(2)

        freqs_cis = freqs_cis[starts_from:starts_from + seq_len]
        freqs_cis = freqs_cis.to(x.device)

        x_reshaped = x.float().view(*x.shape[:-1], self.d_k // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        x_rotated = x_complex * freqs_cis

        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        Abstract method for forward pass of the attention mechanism.
        Subclasses must implement this method.
        """
        pass
    
class MultiHeadAttention(Attention):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__(d_model, num_heads)
        
    def forward(self, x, mask=None, freqs_cis=None, starts_from:int = 0):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if freqs_cis is not None:
            q = self._apply_rope(q, freqs_cis, starts_from=starts_from)
            k = self._apply_rope(k, freqs_cis, starts_from=starts_from)

        # Scaled dot-product attention (Manual implementation commented out for optimization)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # attn_weights = F.softmax(scores, dim=-1)
        # attn_output = torch.matmul(attn_weights, v)

        # Optimized attention computation using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=mask)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        
        return output
    
class CrossAttention(Attention):
    def __init__(self, d_model, num_heads, shift_by_context=False, rope_on_context=False):
        super(CrossAttention, self).__init__(d_model, num_heads)
        assert not (shift_by_context and rope_on_context), "Cannot have both shift_by_context and rope_on_context set to True"
        self.shift_by_context = shift_by_context  # New parameter to control the starting index for RoPE
        self.rope_on_context = rope_on_context  # New parameter to control whether to apply RoPE to the context

    def forward(self, x, context, freqs_cis=None):
        assert not ((self.shift_by_context or self.rope_on_context) and (freqs_cis is None)), "freqs_cis should be provided if either shift_by_context or rope_on_context is True"

        batch_size = x.size(0)

        context_seq_len = context.size(1)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(context).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if freqs_cis is not None:
            q = self._apply_rope(q, freqs_cis, starts_from=context_seq_len * self.shift_by_context) # Treat the input as a continuation of the context sequence, if shift_by_context is True
            if self.rope_on_context:
                k = self._apply_rope(k, freqs_cis)


        # Optimized attention computation using PyTorch's built-in function
        # Note that the mask here should be applied to the cross-attention scores, if provided.
        # Which means it isn't necessarily a square matrix like in self-attention, but rather a mask that aligns with the context sequence length.
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        
        return output