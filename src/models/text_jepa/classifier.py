import torch
from torch import nn
from torch.nn import functional as F

from src.models.text_jepa.modules.encoder import Encoder

class TextJEPAClassifier(nn.Module):
    def __init__(self, encoder_kwargs, num_classes):
        super(TextJEPAClassifier, self).__init__()
        self.encoder = Encoder(**encoder_kwargs)
        self.classifier = nn.Linear(encoder_kwargs['d_model'], num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze encoder parameters

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    
    def train(self, mode=True):
        """
        Keep the encoder in eval mode even while training.
        The purpose of this is to ensure that the encoder's batch norm (if any) and dropout (if any) behave consistently during training.
        """
        super().train(mode)
        self.encoder.eval()
        return self
    
    def forward(self, x):
        with torch.no_grad():
            encoded_repr = self.encoder(x)  # (B, seq_length, d_model)

        x = x.mean(dim=1)  # (B, d_model)

        # Pass through the classifier
        logits = self.classifier(x)  # (B, num_classes)
        return logits