import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.text_jepa.modules.encoder import Encoder # We use the same encoder from text_jepa for the context and target encoders.
from src.models.r_jepa.modules.predictor import CrossAttentionPredictor, CausalAttentionPredictor

class RJEPA(nn.Module):
    def __init__(self, encoder_kwargs, predictor_kwargs, is_cross_attention=True):
        super(RJEPA, self).__init__()
        self.context_encoder = Encoder(**encoder_kwargs)
        self.predictor = CrossAttentionPredictor(**predictor_kwargs) if is_cross_attention else CausalAttentionPredictor(**predictor_kwargs)

        for param in self.context_encoder.parameters():
            param.requires_grad = False  # Ensure context encoder parameters are frozen

        # self.target_encoder = copy.deepcopy(self.context_encoder) # Since the target encoder is a deepcopy, it will also have frozen parameters.
        
    def trainable_parameters(self):
        return list(self.predictor.parameters())
    
    def train(self, mode=True):
        super().train(mode)
        self.context_encoder.eval()
        # self.target_encoder.eval()
        return self
    
    def forward(self, x, reasoning_steps):
        # Context branch: process the input through the context encoder to get context representations.
        context_repr = self.context_encoder(x)  # (B, seq_length, encoder_dim)

        # Target branch: process the reasoning steps through the target encoder to get target representations.
        with torch.no_grad():
            # Reasoning steps is size (B, reasoning_steps, seq_length, d_model)
            # target_repr = self.target_encoder(reasoning_steps)  # (B, reasoning_steps, seq_length, d_model)

            # Since target and context encoders are the same, we can use the context encoder to encode the reasoning steps as well.
            target_repr = self.context_encoder(reasoning_steps)  # (B, reasoning_steps, seq_length, d_model)
            target_repr = target_repr.mean(dim=2)  # (B, reasoning_steps, d_model)

        # Predictor branch: use the context representations to predict the target representations.
        # This is an autoregressive prediction, so we feed in the context and the previous predictions to predict the next step.
        predictions = self.predictor(
            x=target_repr[:, :-1, :], 
            context=context_repr
        )  # (B, reasoning_steps, encoder_dim)

        return predictions, target_repr