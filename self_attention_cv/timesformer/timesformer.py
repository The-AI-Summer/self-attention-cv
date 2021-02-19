import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv.transformer_vanilla import MultiHeadSelfAttention
from self_attention_cv.linformer import LinformerAttention


class TimeSformerBlock(nn.Module):
    def __init__(self, *, frames, patches, dim=512,
                 heads=4, dim_linear_block=1024,
                 activation=nn.GELU,
                 dropout=0.1,
                 linear_spatial_attention=True, k=None):
        """
        Args:
            dim: token's dim
            heads: number of heads
            linear_spatial_attention: if True Linformer-based attention is applied
        """
        super().__init__()
        self.frames = frames
        self.patches = patches

        self.time_att = MultiHeadSelfAttention(dim, heads=heads)

        if linear_spatial_attention:
            if k is None:
                k = self.patches//4 if self.patches//4 > 128 else 128
            proj_shape = (patches, k)
            self.space_att = LinformerAttention(dim, heads=heads, shared_projection=False, proj_shape=proj_shape)
        else:
            self.space_att = MultiHeadSelfAttention(dim, heads=heads)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = rearrange(x, ' b (f p) d -> (b p) f d', f=self.frames)
        x = self.time_att(x) + x
        x = rearrange(x, ' (b p) f d -> (b f) p d', p=self.patches)
        x = self.space_att(x) + x
        x = rearrange(x, '(b f) p d -> b (f p) d', f=self.frames)
        x = self.linear(x) + x
        return x


model = TimeSformerBlock(frames=3, patches=100, dim=512)
a = torch.rand(2, 3 * 100, 512)
b = model(a)
print(b.shape)

# class TimeSformer(nn.Module):
#     def __init__(self, *,
#                  img_dim, frames,
#                  in_channels=3,
#                  patch_dim=16,
#                  num_classes=10,
#                  dim=512,
#                  blocks=6,
#                  heads=4,
#                  dim_linear_block=1024,
#                  dim_head=None,
#                  dropout=0, transformer=None, classification=True):
#         """
#         Adapting ViT for video classification.
#         Best strategy to handle multiple frames so far was
#         Divided Space-Time Attention (T+S). We apply attention to projected
#         image patches, first in time and then in both spatial dims.
#         Args:
#             img_dim: the spatial image size
#             in_channels: number of img channels
#             patch_dim: desired patch dim
#             num_classes: classification task classes
#             dim: the linear layer's dim to project the patches for MHSA
#             blocks: number of transformer blocks
#             heads: number of heads
#             dim_linear_block: inner dim of the transformer linear block
#             dim_head: dim head in case you want to define it. defaults to dim/heads
#             dropout: for pos emb and transformer
#             transformer: in case you want to provide another transformer implementation
#             classification: creates an extra CLS token that we will index in the final classification layer
#         """
#         super().__init__()
#         assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
#         self.p = patch_dim
#         self.classification = classification
#         # tokens = number of img patches * number of frames
#         tokens_spatial = (img_dim // patch_dim) ** 2
#         tokens_spacetime = frames * tokens_spatial
#         self.token_dim = in_channels * (patch_dim ** 2)
#         self.dim = dim
#         self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
#
#         # Projection and pos embeddings
#         self.project_patches = nn.Linear(self.token_dim, dim)
#
#         self.emb_dropout = nn.Dropout(dropout)
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.pos_emb1D = nn.Parameter(torch.randn(tokens_spacetime + 1, dim))
#         # TODO check for layer norm
#         self.mlp_head = nn.Linear(dim, num_classes)
#
#         self.transformer = transformer
#
#     def expand_cls_to_batch(self, batch):
#         """
#         Args:
#             batch: batch size
#         Returns: cls token expanded to the batch size
#         """
#         return self.cls_token.expand([batch, -1, -1])
#
#     def forward(self, vid, mask=None):
#         # Create patches
#         # from [batch, frames, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
#         img_patches = rearrange(vid,
#                                 'b f c (patch_x x) (patch_y y) -> b (f x y) (patch_x patch_y c)',
#                                 patch_x=self.p, patch_y=self.p)
#
#         batch_size, tokens_spacetime, _ = img_patches.shape
#
#         # project patches with linear layer + add pos emb
#         img_patches = self.project_patches(img_patches)
#
#         if self.classification:
#             img_patches = torch.cat((self.expand_cls_to_batch(batch_size), img_patches), dim=1)
#
#         # add pos. embeddings. + dropout
#         # indexing with the current batch's token length to support variable sequences
#         img_patches = img_patches + self.pos_emb1D[:tokens_spacetime + 1, :]
#         patch_embeddings = self.emb_dropout(img_patches)
#
#         # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
#         y = self.transformer(patch_embeddings, mask)
#
#         # we index only the cls token for classification. nlp tricks :P
#         return self.mlp_head(y[:, 0, :]) if self.classification else y
