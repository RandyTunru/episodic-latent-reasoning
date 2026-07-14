import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import CrossAttentionBlock, RMSNorm

class CrossAttentionPredictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, dropout=0.0):
        super(CrossAttentionPredictor, self).__init__()
        self.start_token = nn.Parameter(torch.randn(1, 1, predictor_dim))  # Learnable start token for the predictor

        self.down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, x, context, mask=None, freqs_cis=None):
        x = torch.cat([self.start_token.expand(x.size(0), -1, -1), x], dim=1)  # Prepend start token

        for layer in self.layers:
            x = layer(x, context=context, mask=mask, freqs_cis=freqs_cis)
        x = self.norm(x)
        x = self.up_projection(x)
        return x
