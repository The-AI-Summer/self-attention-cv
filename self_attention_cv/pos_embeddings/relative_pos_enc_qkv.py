import torch
from einops import rearrange
from torch import nn


class Relative2DPosEncQKV(nn.Module):
    def __init__(self, dim_head, dim_v=16, dim_kq=8):
        """
        Implementation of 2D relative positional embeddings for q,v,k
        Out shape shape will be [dim_head, dim, dim]
        Embeddings are shared across heads for all q,k,v
        Based on Axial DeepLab https://arxiv.org/abs/2003.07853

        Args:
            dim_head: the dimension of the head
            dim_v: d_out in the paper
            dim_kq: d_k in the paper
        """
        super().__init__()
        self.dim = dim_head
        self.dim_head_v = dim_v
        self.dim_head_kq = dim_kq
        self.qkv_chan = 2 * self.dim_head_kq + self.dim_head_v

        # 2D relative position embeddings of q,k,v:
        self.relative = nn.Parameter(torch.randn(self.qkv_chan, dim_head * 2 - 1), requires_grad=True)
        self.relative_index_2d = self.relative_index()

    def relative_index(self):
        # integer lists from 0 to 63
        query_index = torch.arange(self.dim).unsqueeze(0)  # [1, dim]
        key_index = torch.arange(self.dim).unsqueeze(1)  # [dim, 1]

        relative_index_2d = (key_index - query_index) + self.dim - 1  # dim X dim
        return rearrange(relative_index_2d, 'i j->(i j)')  # flatten

    def forward(self):
        rel_indx = self.relative_index_2d.to(self.relative.device)
        all_embeddings = torch.index_select(self.relative, 1, rel_indx)  # [head_planes , (dim*dim)]

        all_embeddings = rearrange(all_embeddings, ' c (x y)  -> c x y', x=self.dim)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_kq, self.dim_head_kq, self.dim_head_v],
                                                            dim=0)
        return q_embedding, k_embedding, v_embedding
