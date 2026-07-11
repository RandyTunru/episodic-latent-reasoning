import torch
from torch import nn
from torch.nn import functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU()
        )
        self.up_linear = nn.Linear(d_model, d_ff, bias=False)
        self.down_linear = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate_output = self.gate_linear(x)
        up_output = self.up_linear(x)
        ff_output = gate_output * up_output
        output = self.down_linear(ff_output)
        return output