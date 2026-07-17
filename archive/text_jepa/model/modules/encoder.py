import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, RMSNorm

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, dropout=0.0):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

        # RoPE Frequencies 
        head_dim = d_model // num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        t = torch.arange(max_seq_length) 

        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def forward(self, x, keep_indices=None):
        x = self.token_embedding(x)

        if keep_indices is not None:
            # keep_indices is (B, num_kept_tokens)

            # 1. Gather the sparse token embeddings
            expanded_indices = keep_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.gather(x, dim=1, index=expanded_indices) # (B, num_kept_tokens, d_model)

            # 2. Gather RoPE frequencies for the kept tokens
            # freqs_cis is (max_seq_length, head_dim), we need to gather the frequencies for the kept tokens
            # Since keep_indices is (B, num_kept_tokens), we can use advanced indexing to gather the frequencies
            batch_freqs_cis = self.freqs_cis[keep_indices]  # (B, num_kept_tokens, head_dim)
        else:
            batch_freqs_cis = self.freqs_cis[:x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1)

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=batch_freqs_cis)
        x = self.norm(x)
        return x
