import torch
import torch.nn as nn

from .linformer import LinformerAttention
from ..transformer_vanilla import TransformerBlock


class LinformerBlock(TransformerBlock):
    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1,
                 shared_projection=False, proj_shape=None,
                 trainable_proj=False, activation=nn.GELU):
        super().__init__(dim=dim, dim_linear_block=dim_linear_block, dropout=dropout, activation=activation)
        self.mhsa = LinformerAttention(dim=dim,
                                       heads=heads,
                                       dim_head=dim_head,
                                       shared_projection=shared_projection,
                                       proj_shape=proj_shape,
                                       trainable_proj=trainable_proj)

    def forward(self, x, proj_mat=None):
        return super().forward(x, proj_mat)


class LinformerEncoder(nn.Module):
    def __init__(self, dim, tokens, k=None, blocks=4, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1,
                 shared_projection=True,
                 trainable_proj=True, activation=nn.GELU):
        """
        Based on the Linformer paper
        Link: https://arxiv.org/pdf/2006.04768.pdf

        Args:
            dim: token's dimension, i.e. word embedding vector size
            tokens: sequence length
            blocks: number of sequential blocks
            heads: the number of distinct representations to learn per block
            dim_head: the dim of the head.
            dim_linear_block: reprojection dim. usually multiple of dim
            dropout: dropout in mhsa block
            activation: MHSA block activation

            Specific parameters for Linformer:

            Headwise sharing and Key-value sharing are on by default!

            shared_projection: if the projection matrix E will be shared among layers
            (it will have to be passed in the forward that way)
            trainable_proj: you can enable this option to make it trainable (non-trainable in the paper)
            tokens: (tokens, projection dimension k), where k is the projection dimension of the linformer

            Choice of k for sequences of length n so that
            Linformer’s performance is nearly on par with the original Transformer:
            Practical:
            k = 128 for n = 512
            k = 256 for n = 1024
            Default is n/4 as in most of the paper's experiments
            """
        super().__init__()
        self.shared_projection = shared_projection
        self.k = k if k is not None else tokens // 4
        proj_shape = [tokens, self.k]

        if self.shared_projection:
            self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)

        self.block_list = [LinformerBlock(dim=dim, heads=heads, dim_head=dim_head,
                                          dim_linear_block=dim_linear_block, dropout=dropout,
                                          shared_projection=shared_projection, proj_shape=proj_shape,
                                          trainable_proj=trainable_proj, activation=activation)
                           for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x):
        for layer in self.layers:
            if self.shared_projection:
                x = layer(x, self.E)
            else:
                x = layer(x)
        return x
