import torch
import torch.nn as nn
from einops import rearrange,repeat
from self_attention_cv.pos_embeddings.relative_embeddings_1D import RelPosEmb1D


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        self.h, self.w = feat_map_size # height , width
        self.total_tokens = self.h * self.w
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head,heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head,heads)

    def forward(self, q):
        """
        Args:
            q: [batch, heads, tokens, dim_head]

        Returns:

        """
        # out: [batch head*w h h]
        r_h = self.emb_w(rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        r_h = repeat(r_h, 'b c h w -> b c copy h w', copy=3)

