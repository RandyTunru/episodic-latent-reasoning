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

    def forward(self, x, keep_indices=None):
        x = self.down_projection(x)
        for layer in self.layers:
            x = layer(x, keep_indices=keep_indices)
        x = self.norm(x)
        x = self.up_projection(x)
        return x
