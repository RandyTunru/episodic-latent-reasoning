import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, RMSNorm

class Predictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, dropout=0.0):
        super(Predictor, self).__init__()
        self.down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, x, freqs_cis=None, keep_indices=None):
        x = self.down_projection(x)

        # keep_indices should be implemented here

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=freqs_cis)
        x = self.norm(x)
        x = self.up_projection(x)
        return x
