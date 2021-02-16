import torch.nn as nn
from einops import rearrange

from self_attention_cv.pos_embeddings.relative_embeddings_1D import RelPosEmb1D


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        """
        Based on Bottleneck transformer paper
        paper: https://arxiv.org/abs/2101.11605 . Figure 4
        Output: qr^T [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        self.h, self.w = feat_map_size  # height , width
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads)

    def expand_emb(self, r, dim_size):
        # Decompose and unsqueeze dimension
        r = rearrange(r, 'b (h x) i j -> b h x () i j', x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1, -1]  # -1 indicates no expansion
        r = r.expand(expand_index)
        return rearrange(r, 'b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)')

    def forward(self, q):
        """
        Args:
            q: [batch, heads, tokens, dim_head]
        Returns: [ batch, heads, tokens, tokens]
        """
        assert self.total_tokens == q.shape[2], f'Tokens {q.shape[2]} of q must \
        be equal to the product of the feat map size {self.total_tokens} '

        # out: [batch head*w h h]
        r_h = self.emb_w(rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.h)
        return q_r  # q_r transpose in figure 4 of the paper
