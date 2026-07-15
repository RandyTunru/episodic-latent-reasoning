import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, RMSNorm

class Predictor(nn.Module):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, max_seq_length, dropout=0.0):
        super(Predictor, self).__init__()
        self.predictor_dim = predictor_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))  # Learnable mask token

        self.down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)  # Initialize the mask token

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
        """
        Args:
            context_tokens: (B, context_len, d_model)
            context_indices: (B, context_len)
            target_indices: (B, num_targets, target_len)
        """
        x = self.down_projection(context_tokens)

        # context_indices and target_indices should be implemented here
        expanded_context_indices = context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)  # (B, context_len, d_model)
        context_freqs_cis = torch.gather(self.freqs_cis, dim=0, index=expanded_context_indices)  # (B, context_len, head_dim)

        num_context = context_tokens.size(1)
        context_tokens = torch.repeat_interleave(context_tokens, repeats=target_indices.size(1), dim=0)  # (B * num_targets, context_len, d_model)
        
        batch_size, num_blocks, block_size = target_indices.shape

        flat_target_indices = target_indices.reshape(batch_size * num_blocks, block_size)  # (B * num_targets, target_len)
        mask_tokens = self.mask_token.expand(batch_size * num_blocks, block_size, -1)  # (B * num_targets, target_len, d_model)

        expanded_target_indices = flat_target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)  # (B * num_targets, target_len, d_model)
        target_freqs_cis = torch.gather(self.freqs_cis, dim=0, index=expanded_target_indices)  # (B * num_targets, target_len, head_dim)

        x = torch.cat([context_tokens, mask_tokens], dim=1)  # (B * num_targets, context_len + target_len, d_model)
        batch_freqs_cis = torch.cat([context_freqs_cis, target_freqs_cis], dim=1)  # (B * num_targets, context_len + target_len, head_dim)

        for layer in self.layers:
            x = layer(x, mask=None, freqs_cis=batch_freqs_cis)
        x = self.norm(x)

        predictions = x[:, num_context:, :]  # (B * num_targets, target_len, d_model)

        predictions = self.up_projection(predictions)

        return predictions.view(batch_size, num_blocks, block_size, -1)  # (B, num_targets, target_len, d_model)
