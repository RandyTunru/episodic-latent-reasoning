from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.block import SelfAttentionBlock, CrossAttentionBlock, RMSNorm

class RJEPAPredictor(nn.Module, ABC):
    def __init__(self, encoder_dim, predictor_dim, num_heads, max_seq_length):
        super(RJEPAPredictor, self).__init__()
        self.start_token = nn.Parameter(torch.randn(1, 1, predictor_dim))  # Learnable start token for the predictor

        self.context_down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.reasoning_down_projection = nn.Linear(encoder_dim, predictor_dim)
        self.norm = RMSNorm(predictor_dim)
        self.up_projection = nn.Linear(predictor_dim, encoder_dim)

        # RoPE Frequencies 
        head_dim = predictor_dim // num_heads
        freqs = torch.arange(0, head_dim, 2) / head_dim
        freqs = 1 / (10000 ** freqs)

        t = torch.arange(max_seq_length) 

        angles = torch.outer(t, freqs)  
        freqs_cis = torch.polar(torch.ones_like(angles), angles)  
        
        # Causal Mask 
        mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).bool().unsqueeze(0).unsqueeze(0)
        
        # Register them as buffers to ensure they are moved to the correct device with the model
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.register_buffer('mask', mask, persistent=False)

    @abstractmethod
    def forward(self, x, context=None):
        pass

class CrossAttentionPredictor(RJEPAPredictor):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, max_seq_length, dropout=0.0):
        super(CrossAttentionPredictor, self).__init__(encoder_dim, predictor_dim, num_heads, max_seq_length)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, context):
        # Project context to predictor dimension
        context = self.context_down_projection(context) #(B, context_len, predictor_dim)
        # Project input to predictor dimension
        x = self.reasoning_down_projection(x) #(B, reasoning_steps, predictor_dim)

        x = torch.cat([self.start_token.expand(x.size(0), -1, -1), x], dim=1)  # Prepend start token

        for layer in self.layers:
            x = layer(x, context=context, mask=self.mask, freqs_cis=self.freqs_cis)
        x = self.norm(x)

        x = self.up_projection(x)
        return x

class CausalAttentionPredictor(RJEPAPredictor):
    def __init__(self, encoder_dim, predictor_dim, num_heads, d_ff, num_layers, max_seq_length, dropout=0.0):
        super(CausalAttentionPredictor, self).__init__(encoder_dim, predictor_dim, num_heads, max_seq_length)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(predictor_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, context):
        context_len = context.size(1)

        # Project context to predictor dimension
        context = self.context_down_projection(context)  #(B, context_len, predictor_dim)
        # Project input to predictor dimension
        x = self.reasoning_down_projection(x) #(B, reasoning_steps, predictor_dim)

        x = torch.cat([context, self.start_token.expand(x.size(0), -1, -1), x], dim=1)  # Prepend context and start token

        for layer in self.layers:
            x = layer(x, mask=self.mask, freqs_cis=self.freqs_cis)  
        x = self.norm(x)

        x = self.up_projection(x)
        # Return only the predictions corresponding to the reasoning steps
        # start with context_len because slicing is inclusive of the start index, 
        # context_len would be the index of the output corresponding the start_token, which is our first reasoning step.
        return x[:, context_len:, :]  