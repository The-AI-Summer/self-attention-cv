import torch
from einops import rearrange
from torch import nn

from self_attention_cv.linformer.linformer import project_vk_linformer
from self_attention_cv.transformer_vanilla.mhsa import compute_mhsa
from ..common import expand_to_batch


def split_cls(x):
    """
    split the first token of the sequence in dim 2
    """
    # Note by indexing the first element as 0:1 the dim is kept in the tensor's shape
    return x[:, :, 0:1, ...], x[:, :, 1:, ...]


def time_att_rearrange(x, frames):
    return rearrange(x, 'b h (f p) d -> (b p) h f d', f=frames)


def space_att_rearrange(x, patches):
    return rearrange(x, 'b h (f p) d -> (b f) h p d', p=patches)


def merge_timespace(x, batch, space=False):
    out_indices = 'b h (k t) d' if space else 'b h (t k) d'
    return rearrange(x, f'(b k) h t d -> {out_indices}', b=batch)


class SpacetimeMHSA(nn.Module):
    def __init__(self, dim, tokens_to_attend, space_att, heads=8,
                 dim_head=None, classification=True,
                 linear_spatial_attention=False, k=None):
        """
        Attention through time and space to process videos
        choose mode (whether to operate in space and time with space_att (bool) )
        CLS token is used for video classification, which will attend all tokens in both
        space and time before attention only in time or space.

        Code is based on lucidrains repo: https://github.com/lucidrains/TimeSformer-pytorch
        Args:
            dim: token's dimension, i.e. word embedding vector size
            tokens_to_attend: space (patches) or time (frames) tokens that we will attend to
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head=dim//heads.
            space_att: whether to use space or time attention in this block
            classification: when True a classification token is expected in the forward call
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.space_att = space_att
        self.reshape_timespace = space_att_rearrange if self.space_att else time_att_rearrange
        self.tokens_to_attend = tokens_to_attend
        self.classification = classification
        self.linear_spatial_attention = linear_spatial_attention and self.space_att
        self.k = k if k is not None else 256

        if self.linear_spatial_attention:
            proj_shape = tuple((self.tokens_to_attend + 1, k))
            self.E = torch.nn.Parameter(torch.randn(proj_shape))

    def forward(self, x):
        """
        Expects input x with merged tokens in both space and time
        Args:
            x: [batch, tokens_timespace+ cls_token, dim*3*heads ]
        """
        assert x.dim() == 3
        batch, token_dim = x.shape[0], 2
        qkv = self.to_qvk(x)

        # decomposition to q,v,k and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        if self.classification:
            # only the cls token will to ALL tokens in both space+time: tokens_timespace
            (cls_q, q_3D) = split_cls(q)
            out_cls = compute_mhsa(cls_q, k, v, scale_factor=self.scale_factor)

            # reshape for space or time attention here only
            (cls_k, k_3D), (cls_v, v_3D) = map(split_cls, (k, v))

            # this is where we decompose/separate the tokens for attention in time or space only
            q_sep, k_sep, v_sep = map(self.reshape_timespace, [q_3D, k_3D, v_3D], [self.tokens_to_attend] * 3)

            # we have to expand/repeat the cls_k, and cls_v to k,v
            cls_k, cls_v = map(expand_to_batch, (cls_k, cls_v), (k_sep.shape[0], v_sep.shape[0]))

            k = torch.cat((cls_k, k_sep), dim=token_dim)
            v = torch.cat((cls_v, v_sep), dim=token_dim)

            if self.linear_spatial_attention:
                v, k = project_vk_linformer(v, k, self.E)

            # finally the conventional attention only through space/time
            out_mhsa = compute_mhsa(q_sep, k, v, scale_factor=self.scale_factor)

            # merge tokens from space and time
            out_mhsa = merge_timespace(out_mhsa, batch, self.space_att)
            # and spacetime cls token
            out = torch.cat((out_cls, out_mhsa), dim=token_dim)
        else:
            out = compute_mhsa(q, k, v)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
