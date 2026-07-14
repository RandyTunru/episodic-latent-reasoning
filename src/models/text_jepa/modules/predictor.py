import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, RMSNorm

class Predictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, max_seq_length, dropout=0.0):
        super(Predictor, self).__init__()
        self.down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

        # RoPE Frequencies 
        head_dim = predictor_dim // num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        t = torch.arange(max_seq_length) 

        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def forward(self, context_tokens, context_indices, target_indices):
        x = self.down_projection(context_tokens)

        # context_indices and target_indices should be implemented here

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=self.freqs_cis)
        x = self.norm(x)
        x = self.up_projection(x)
        return x
