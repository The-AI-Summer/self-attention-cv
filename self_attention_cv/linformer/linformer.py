import torch
from einops import rearrange
from torch import nn

from ..transformer_vanilla.mhsa import compute_mhsa


def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k


class LinformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, shared_projection=True, proj_shape=None, trainable_proj=True):
        """
        Based on the Linformer paper
        Link: https://arxiv.org/pdf/2006.04768.pdf

        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head.

            shared_projection: if the projection matrix will be shared among layers
            (it will have to be passed in the forward that way)
            trainable_proj: if the projection matrix E matrix is not shared,
            you can enable this option to make it trainable (non trainable in the paper)
            proj_shape: 2-tuple (tokens,k), where k is the projection dimension of the linformer
            """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.shared_projection = shared_projection

        if not shared_projection:
            self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)
            self.k = proj_shape[1]

    def forward(self, x, proj_mat=None):
        assert x.dim() == 3
        E = proj_mat if (self.shared_projection and proj_mat is not None) else self.E
        assert x.shape[1] == E.shape[0], f'{x.shape[1]} Token in the input sequence while' \
                                         f' {E.shape[0]} were provided in the E proj matrix'

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        v, k = project_vk_linformer(v, k, E)

        out = compute_mhsa(q, k, v, scale_factor=self.scale_factor)
        # re-compose: merge heads with dim_head

        out = rearrange(out, "b h i d -> b i (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
