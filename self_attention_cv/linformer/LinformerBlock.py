import torch.nn as nn
from ..transformer_vanilla import TransformerBlock, TransformerEncoder
from .linformer import LinformerAttention


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
        super().forward(x, proj_mat)

# class LinformerEncoder(TransformerEncoder):
#     def __init__(self, dim, heads=8, dim_head=None,
#                  dim_linear_block=1024, dropout=0.1,
#                  shared_projection=False, proj_shape=None,
#                  trainable_proj=False, activation=nn.GELU):
#         super().__init__()
