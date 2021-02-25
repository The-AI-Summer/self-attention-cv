import numpy as np
import torch
from einops import rearrange
from torch import nn


class TiedRowAxialAttention(nn.Module):
    def __init__(self, *, dim, rows, heads=8, dim_head=None):
        """
        Tied row attention uses a single attention map for all tokens in the MSA
        Applies tied attention by decomposing batches*rows and summing Q*K^T
        over all rows. batches*rows is recomposed by multipling the attention weights back
        with the value vector
        The  Equation 1 in the paper.
        Tied attention reduces the memory footprint of the row attentions
        from O(rows dim^2) to O(dim^2).
        Link: https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1
        Args:
            dim: token's dimension i.e column pixels of an image
            rows: number of rows with shared/tied attention that will be summed in Q*K^T
            heads: the number of distinct representations to learn
            dim_head: the dim of the head.
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.rows = rows
        self.scale_factor = (self.rows * self.dim_head) ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3, 'Ensure your input is 4D: [b * width, chan, height] or [b * height, chan, width]'
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decompose heads, (q,v,k) tuple , and rows
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, '(b rows) t (d k h ) -> k b rows h t d ', k=3, h=self.heads, rows=self.rows))

        # resulted shape will be: [batch, heads, tokens, tokens]. Notice that the row index is vanished
        # meaning that this dimension is summed
        scaled_dot_prod = torch.einsum('b r h i d , b r h j d -> b h i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        # calc result per head and row for v
        out = torch.einsum('b h i j , b r h j d -> b r h i d', attention, v)
        # re-compose: merge heads with dim_head
        out = rearrange(out, "b rows h t d -> (b rows) t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
