import torch
from einops import rearrange
from torch import nn


class RelativePosEncQKV(nn.Module):
    """
    Implementation of 1D relative positional embeddings for q,v,k
    resulting shape will be [dim_head, dim, dim] ,
    Embeddings are shared across heads for q,k,v
    Based on Axial DeepLab https://arxiv.org/abs/2003.07853
    """
    def __init__(self, dim, heads_planes, dim_head_v=16, dim_head_kq=8):
        super().__init__()
        self.dim = dim
        self.heads_planes = heads_planes
        self.dim_head_v = dim_head_v
        self.dim_head_kq = dim_head_kq

        # 1D relative position embedding matrix
        self.relative = nn.Parameter(torch.randn(self.heads_planes * 2, dim * 2 - 1), requires_grad=True)
        self.flatten_index = self.find_flattend_index()

    def find_flattend_index(self):
        # integer lists from 0 to 63
        query_index = torch.arange(self.dim).unsqueeze(0) # [1, dim]
        key_index = torch.arange(self.dim).unsqueeze(1)   # [dim, 1]

        relative_index = key_index - query_index + self.dim - 1  # dim X dim
        return rearrange(relative_index, 'i j->(i j)')  # flatten

    def forward(self):
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index)  # [head_planes , (dim*dim)]

        all_embeddings = rearrange(all_embeddings, ' c (x y)  -> c x y',x=self.dim)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_kq, self.dim_head_kq, self.dim_head_v],
                                                            dim=0)
        return q_embedding, k_embedding, v_embedding
