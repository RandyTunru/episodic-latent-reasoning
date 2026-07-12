import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.text_jepa.modules.encoder import Encoder
from src.models.text_jepa.modules.predictor import Predictor

class TextJEPA(nn.Module):
    def __init__(self, encoder_kwargs, predictor_kwargs):
        super(TextJEPA, self).__init__()
        self.context_encoder = Encoder(**encoder_kwargs)
        self.predictor = Predictor(**predictor_kwargs)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Freeze target encoder parameters

    def trainable_parameters(self):
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())
    
    def train(self, mode=True):
        """
        Keep the target encoder in eval mode even while training.
        The purpose of this is to ensure that the target encoder's batch norm (if any) and dropout (if any) behave consistently during training.
        We can't make let the target encoder have an active dropout because the target encoder is used to generate the targets for the predictor, 
        And we want those targets to be deterministic and not stochastic.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self
    
    def forward(self, x, context_indices, target_indices):
        pass