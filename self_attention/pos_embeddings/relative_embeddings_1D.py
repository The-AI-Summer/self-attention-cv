import torch
from einops import rearrange
from torch import einsum


def relative_to_absolute(logits, dim, axis=-1):
    # integer lists from 0 to 63
    query_index = torch.arange(dim).unsqueeze(0)  # [1, dim]
    key_index = torch.arange(dim).unsqueeze(1)  # [dim, 1]

    relative_index = (key_index - query_index) + dim - 1  # dim X dim
    flatten_index = rearrange(relative_index, 'i j->(i j)')  # flatten
    abs_emb = torch.index_select(logits, axis, flatten_index)  # [head_planes , (dim*dim)]
    return rearrange(abs_emb, 'b h t (x y) -> b h t x y', x=dim)


def relative_emb_1d(q, rel_k):
    """

    """
    b, heads, tokens, dim = q.shape
    emb = einsum('b h t d, r d -> b h t r', q, rel_k)
    print(emb.shape)
    emb = relative_to_absolute(emb, dim=dim, axis=-1)
    print(emb.shape)
    return emb


# todo class file and test
q = torch.rand(2, 3, 128, 64)
rel = torch.rand(128 * 2 - 1, 64)
y = relative_emb_1d(q, rel)
print(y.shape)

