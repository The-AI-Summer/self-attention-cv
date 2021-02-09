import torch
import torch.nn as nn
from einops import rearrange


def relative_to_absolute(q_rel, tokens, axis=-1):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
    """
    query_index = torch.arange(tokens).unsqueeze(0)  # [1, dim]
    key_index = torch.arange(tokens).unsqueeze(1)  # [dim, 1]

    relative_index = (key_index - query_index) + tokens - 1  # dim X dim (zero indexed)
    flatten_index = rearrange(relative_index, 'i j->(i j)')  # flatten
    abs_emb = torch.index_select(q_rel, axis, flatten_index)  # [head_planes , (dim*dim)]
    return rearrange(abs_emb, 'b h t (x y) -> b h t x y', x=tokens)





def rel_pos_emb_1d(q, rel_emb):
    """
    Same functionality as RelPosEmb1D
    q shape [batch, heads, tokens, dim]
    rel_emb shape [ 2*tokens-1 , dim]
    """
    tokens = q.shape[2]


    emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    emb = relative_to_absolute(emb, tokens=tokens, axis=-1)
    return emb


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim):
        """
        Output: [batch head tokens tokens]
        Args:
            dim_head: the size of the last dimension of q
        """
        super().__init__()
        scale = dim ** -0.5
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb)
