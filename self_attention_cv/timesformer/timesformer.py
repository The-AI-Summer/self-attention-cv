import torch
import torch.nn as nn
from einops import rearrange

from .spacetime_attention import SpacetimeMHSA
from ..common import expand_to_batch

class TimeSformerBlock(nn.Module):
    def __init__(self, *, frames, patches, dim=512,
                 heads=None, dim_linear_block=1024,
                 activation=nn.GELU,
                 dropout=0.1, classification=True,
                 linear_spatial_attention=False, k=None):
        """
        Args:
            dim: token's dim
            heads: number of heads
            linear_spatial_attention: if True Linformer-based attention is applied
        """
        super().__init__()
        self.frames = frames
        self.patches = patches
        self.classification = classification

        self.time_att = nn.Sequential(nn.LayerNorm(dim),
                                      SpacetimeMHSA(dim, tokens_to_attend=self.frames, space_att=False,
                                                    heads=heads, classification=self.classification))

        self.space_att = nn.Sequential(nn.LayerNorm(dim),
                                       SpacetimeMHSA(dim, tokens_to_attend=self.patches, space_att=True,
                                                     heads=heads, classification=self.classification,
                                                     linear_spatial_attention=linear_spatial_attention, k=k))

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        assert x.dim() == 3
        x = self.time_att(x) + x
        x = self.space_att(x) + x
        x = self.mlp(x) + x
        return x


class Timesformer(nn.Module):
    def __init__(self, *,
                 img_dim, frames,
                 num_classes=None,
                 in_channels=3,
                 patch_dim=16,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 activation=nn.GELU,
                 dropout=0,
                 linear_spatial_attention=False, k=None):
        """
        Adapting ViT for video classification.
        Best strategy to handle multiple frames so far is
        Divided Space-Time Attention (T+S). We apply attention to projected
        image patches, first in time and then in both spatial dims.
        Args:
            img_dim: the spatial image size
            frames: video frames
            num_classes: classification task classes
            in_channels: number of img channels
            patch_dim: desired patch dim

            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            linear_spatial_attention: for Linformer linear attention
            k: for Linformer linear attention
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        # classification: creates an extra CLS token that we will index in the final classification layer
        self.classification = True if num_classes is not None else False
        img_patches = (img_dim // patch_dim) ** 2
        # tokens = number of img patches * number of frames
        tokens_spacetime = frames * img_patches
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens_spacetime + 1, dim))

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        transormer_blocks = [TimeSformerBlock(
            frames=frames, patches=img_patches, dim=dim,
            heads=heads, dim_linear_block=dim_linear_block,
            activation=activation,
            dropout=dropout,
            linear_spatial_attention=linear_spatial_attention, k=k)
            for _ in range(blocks)]

        self.transformer = nn.Sequential(*transormer_blocks)

    def forward(self, vid):
        # Create patches as in ViT wherein frames are merged with patches
        # from [batch, frames, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(vid,
                                'b f c (patch_x x) (patch_y y) -> b (f x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)

        batch_size, tokens_spacetime, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, batch_size), img_patches), dim=1)
        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens_spacetime + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y
