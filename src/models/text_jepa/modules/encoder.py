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

        # keep_indices should be implemented here

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=self.freqs_cis)
        x = self.norm(x)
        return x
