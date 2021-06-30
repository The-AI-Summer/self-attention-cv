import torch
import torch.nn as nn
from ..common import expand_to_batch


# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncodingSin(nn.Module):

    def __init__(self, dim, dropout=0.1, max_tokens=5000):
        super(PositionalEncodingSin, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.Tensor([10000.0])) / dim))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe

    def forward(self, x):
        batch, seq_tokens, _ = x.size()
        x = x + expand_to_batch( self.pe[:, :seq_tokens, :], desired_size=batch)
        return self.dropout(x)