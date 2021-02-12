import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=64,
                 dropout=0, transformer=None):
        """
        Minimal reimplementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        # tokens = number of patches
        tokens = (img_dim // patch_dim) ** 2
        token_dim = in_channels * patch_dim ** 2

        # Projection and pos embeddings
        self.project_patches = nn.Linear(token_dim, dim)
        self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))
        self.emb_dropout = nn.Dropout(dropout)

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # project patches with linear layer + add pos emb
        patch_embeddings = self.emb_dropout(self.project_patches(img_patches) + self.pos_emb1D)

        # feed patch_embeddings to transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # mean dim + final linear layer
        return self.mlp_head(y.mean(dim=1))


