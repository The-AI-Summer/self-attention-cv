import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv.linformer import LinformerAttention
from self_attention_cv.transformer_vanilla import MultiHeadSelfAttention


def split_cls(x):
    """
    split the first token of the sequence in dim 1
    """
    # Note by indexing the first element as 0:1 the dim is kept in the tensor's shape
    return x[:, 0:1, ...], x[:, 1:, ...]

def expand_cls_to_batch(cls_token, desired_batch_dim):
    """
    Args:
        cls_token: batch tokens dim
        batch: batch size
    Returns: cls token expanded to the batch size
    """
    expand_times = desired_batch_dim//cls_token.shape[0]
    return cls_token.expand([expand_times, -1, -1])

class TimeSformerBlock(nn.Module):
    def __init__(self, *, frames, patches, dim=512,
                 heads=None, dim_linear_block=1024,
                 activation=nn.GELU,
                 dropout=0.1,
                 linear_spatial_attention=True, k=None, cls_token=True):
        """
        Args:
            dim: token's dim
            heads: number of heads
            linear_spatial_attention: if True Linformer-based attention is applied
        """
        super().__init__()
        self.frames = frames
        self.patches = patches
        self.cls_token = 1 if cls_token else 0

        self.time_att = MultiHeadSelfAttention(dim, heads=heads)

        if linear_spatial_attention:
            if k is None:
                k = self.patches // 4 if self.patches // 4 > 128 else 128
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

    def forward(self, x_inp):
        assert x_inp.dim() == 3
        cls, x = split_cls(x_inp)
        x = rearrange(x, 'b (f p) d -> (b p) f d', f=self.frames)

        cls = expand_cls_to_batch(cls, x.shape[0])
        x = torch.cat((cls, x), dim=1)
        y = self.time_att(x) + x
        cls, y = split_cls(y)
        y = rearrange(y, '(b p) f d -> (b f) p d', p=self.patches)
        cls = expand_cls_to_batch(cls, y.shape[0])
        #y = torch.cat((cls, y), dim=1)
        y = self.space_att(y) + y
        y = rearrange(y, '(b f) p d -> b (f p) d', f=self.frames)
        y = self.linear(y) + y
        return y


class Timesformer(nn.Module):
    def __init__(self, *,
                 img_dim, frames,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 activation=nn.GELU,
                 dropout=0,
                 linear_spatial_attention=True, k=None,
                 classification=True):
        """
        Adapting ViT for video classification.
        Best strategy to handle multiple frames so far was
        Divided Space-Time Attention (T+S). We apply attention to projected
        image patches, first in time and then in both spatial dims.
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        self.classification = classification
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
        # TODO check for layer norm
        self.mlp_head = nn.Linear(dim, num_classes)

        transormer_blocks = nn.ModuleList([TimeSformerBlock(frames=frames, patches=img_patches, dim=dim,
                                                            heads=heads, dim_linear_block=dim_linear_block,
                                                            activation=activation,
                                                            dropout=dropout,
                                                            linear_spatial_attention=linear_spatial_attention, k=k) for
                                           _
                                           in range(blocks)])
        self.transformer = nn.Sequential(*transormer_blocks)



    def forward(self, vid):
        # Create patches
        # from [batch, frames, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(vid,
                                'b f c (patch_x x) (patch_y y) -> b (f x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)

        batch_size, tokens_spacetime, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_cls_to_batch(self.cls_token, batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens_spacetime + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y
