from ..transformer_vanilla import MultiHeadSelfAttention, TransformerEncoder

import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv import TransformerEncoder


class Transformer3dSeg(nn.Module):
    def __init__(self, *,
                 subvol_dim=24,  # W in the paper
                 patch_dim=8,
                 num_classes = 2,
                 in_channels=3,
                 dim=1024,
                 blocks=7,
                 heads=4,
                 dim_linear_block=1024,
                 dropout=0, transformer=None):
        """
        Replication/re-implementation based on the paper:
        Convolution-Free Medical Image Segmentation using Transformers
        Paper: https://arxiv.org/abs/2102.13645
        The model accepts a sub-volume of subvol_dim X subvol_dim X subvol_dim, and return a cube segmentation
        of n = subvol_dim//patch_dim

        Args:
            subvol_dim: cube block size. you have to sample the original 3D volume
            patch_dim: the desired cuded 3D patch dimension
            num_classes: segmentation classes
            in_channels: channels/modalities
            dim: the projected dim of the patches that will be the input dim of the transformer
            blocks: number of repreated transformer blocks
            heads: number of heads
            dim_linear_block: MLP dim inside Transformer block
            dropout: dropout for the transformer block
            transformer: if provided another transformer is used. Vanilla Transformer by default as in ViT
        """
        super().__init__()
        self.p = patch_dim
        self.num_classes = num_classes
        # tokens = number of 3D patches
        self.n = subvol_dim//patch_dim
        self.tokens = self.n ** 3

        self.mid_token = (self.tokens//2)

        self.token_dim = in_channels * (self.p ** 3)
        self.dim = dim

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.pos_emb1D = nn.Parameter(torch.randn(self.tokens, dim))

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

        self.mlp_seg_head = nn.Linear(dim, self.tokens * self.num_classes)

    def forward(self, img, mask=None):
        """
        Args:
            img: square subvolume sub-sampled from the 3D image  b,c,W,W,W
            mask: mask must be supposrted also
        Returns: center patch segmentation
        """
        # Create patches
        # from [batch, channels, h, w ,z ] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) (patch_z z) -> b (x y z) (patch_x patch_y patch_z c)',
                                patch_x=self.p, patch_y=self.p, patch_z=self.p)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)
        # take mid token only and take final segmentation for self.n ** 3 sub-sub-volume :)
        y = self.mlp_seg_head(y[:, self.mid_token, :])

        return rearrange(y,'b (x y z classes) -> b classes x y z',x=self.n, y=self.n, z=self.n )
