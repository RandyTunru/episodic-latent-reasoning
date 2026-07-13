import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, RMSNorm

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            SelfAttentionBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, freqs_cis=None, keep_indices=None):
        x = self.token_embedding(x)

        # keep_indices should be implemented here

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=freqs_cis)
        x = self.norm(x)
        return x
