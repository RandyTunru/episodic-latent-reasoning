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
        batch_size, num_blocks, block_size = target_indices.shape

        # Context branch: only the context patches through the context encoder.
        context_repr = self.context_encoder(x, keep_indices=context_indices)

        # Target branch: full sequence through the EMA encoder, no gradients.
        with torch.no_grad():
            target_full = self.target_encoder(x)  # (B, num_patches, encoder_dim)

            flat_indices = target_indices.reshape(batch_size, num_blocks * block_size)
            expanded = flat_indices.unsqueeze(-1).expand(-1, -1, target_full.size(-1))
            targets = torch.gather(target_full, dim=1, index=expanded)
            targets = targets.view(batch_size, num_blocks, block_size, -1)

        # Predictor branch: extends the context representation to predict the target patches.
        predictions = self.predictor(context_repr, context_indices, target_indices)
        return predictions, targets
    
    @torch.no_grad()
    def update_target_encoder(self, momentum=0.999):
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)