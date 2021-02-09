import torch
from torch import nn, einsum


class AbsPosEmb1D(nn.Module):
    """
    Given query q of shape [batch heads tokens dim] we multiply
    q by all the flattened absolute differences between tokens.
    Learned embedding representations are shared across heads
    """

    def __init__(self, tokens, dim):
        """
        Output: [batch head tokens tokens]
        Args:
            dim_head: the size of the last dimension of q
        """
        super().__init__()
        scale = dim ** -0.5
        self.abs_pos_emb = nn.Parameter(torch.randn(tokens, dim) * scale)

    def forward(self, q):
        return einsum('b h i d, j d -> b h i j', q, self.abs_pos_emb)