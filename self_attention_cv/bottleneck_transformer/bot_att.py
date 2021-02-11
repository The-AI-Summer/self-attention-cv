import torch
from einops import rearrange
from torch import nn, einsum

from ..pos_embeddings import RelPosEmb2D


class BottleneckAttention(nn.Module):
    def __init__(
            self,
            dim,
            fmap_size,
            heads=4,
            dim_head=None,
            content_positional_embedding=True
    ):
        """
        tensorflow code gist
        https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
        lucidrains' code that I also studied until I finally understood how it works:
        https://github.com/lucidrains/bottleneck-transformer-pytorch
        paper: https://arxiv.org/abs/2101.11605
        Args:
            dim: dimension of the token vector (d_model)
            fmap_size: tuple with the feat map spatial dims
            heads: number of heads representations
            dim_head: inner dimension of the head. dim / heads by default
            content_positional_embedding: whether to include the 2 rel. pos embedding for the query
        """
        super().__init__()
        self.heads = heads
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.scale = self.dim_head ** -0.5
        self.fmap_size = fmap_size
        self.content_positional_embedding = content_positional_embedding

        self.to_qkv = nn.Conv2d(dim, heads * self.dim_head * 3, 1, bias=False)

        self.height = self.fmap_size[0]
        self.width = self.fmap_size[1]

        if self.content_positional_embedding:
            self.pos_emb2D = RelPosEmb2D(feat_map_size=fmap_size, dim_head=self.dim_head)

    def forward(self, x):
        assert x.dim() == 4, f'Expected 4D tensor, got {x.dim()}D tensor'

        # [batch (heads*3*dim_head) height width]
        qkv = self.to_qkv(x)
        # decompose heads and merge spatial dims as tokens
        q, k, v = tuple(rearrange(qkv, 'b (d k h ) x y  -> k b h (x y) d', k=3, h=self.heads))

        # i, j refer to tokens
        dot_prod = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.content_positional_embedding:
            dot_prod = dot_prod + self.pos_emb2D(q)

        attention = torch.softmax(dot_prod, dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        # Merge heads and decompose tokens to spatial dims
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=self.height, y=self.width)
        return out
