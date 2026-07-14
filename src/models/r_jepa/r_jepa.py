import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.text_jepa.modules.encoder import Encoder # We use the same encoder from text_jepa for the context and target encoders.
from src.models.r_jepa.modules.predictor import CrossAttentionPredictor

class RJEPA(nn.Module):
    def __init__(self, encoder_kwargs, predictor_kwargs):
        super(RJEPA, self).__init__()
        self.context_encoder = Encoder(**encoder_kwargs)
        self.predictor = CrossAttentionPredictor(**predictor_kwargs)

        for param in self.context_encoder.parameters():
            param.requires_grad = False  # Ensure context encoder parameters are frozen

        self.target_encoder = copy.deepcopy(self.context_encoder) # Since the target encoder is a deepcopy, it will also have frozen parameters.

        # RoPE Frequencies 
        head_dim = self.d_model // self.num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        scale_factor = self.max_seq_len 

        t = torch.arange(self.max_seq_len) / scale_factor # Divide by scale_factor to adjust the effective sequence length for RoPE

        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Causal Mask 
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).bool().unsqueeze(0).unsqueeze(0)
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.register_buffer('mask', mask, persistent=False)
        
    def trainable_parameters(self):
        return list(self.predictor.parameters())
    
    def train(self, mode=True):
        super().train(mode)
        self.context_encoder.eval()
        self.target_encoder.eval()
        return self
    
    def forward(self, x, reasoning_steps):
        # Context branch: process the input through the context encoder to get context representations.
        context_repr = self.context_encoder(x)  # (B, seq_length, encoder_dim)

        # Target branch: process the reasoning steps through the target encoder to get target representations.
        with torch.no_grad():
            # Reasoning steps is size (B, reasoning_steps, seq_length, d_model)
            target_repr = self.target_encoder(reasoning_steps)  # (B, reasoning_steps, encoder_dim)

        # Predictor branch: use the context representations to predict the target representations.
        # This is an autoregressive prediction, so we feed in the context and the previous predictions to predict the next step.
        predictions = self.predictor(
            x=reasoning_steps[:, :-1, :], 
            context=context_repr, 
            mask=self.mask, 
            freqs_cis=self.freqs_cis
        )  # (B, reasoning_steps, encoder_dim)

        return predictions, target_repr