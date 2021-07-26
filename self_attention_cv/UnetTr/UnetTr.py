import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv.UnetTr.modules import TranspConv3DBlock, BlueBlock, Conv3DBlock
from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
from self_attention_cv.transformer_vanilla import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        # makes TransformerBlock device aware
        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(TransformerBlock(dim=embed_dim, heads=num_heads,
                                                    dim_linear_block=dim_linear_block, dropout=dropout, prenorm=True))

    def forward(self, x):
        extract_layers = []
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
            if (depth + 1) in self.extract_layers:
                extract_layers.append(x)

        return extract_layers


# based on https://arxiv.org/abs/2103.10504
# implementation is influenced by practical details missing in the paper that can be found
# https://github.com/Project-MONAI/MONAI/blob/027947bf91ff0dfac94f472ed1855cd49e3feb8d/monai/networks/nets/unetr.py
class UNETR(nn.Module):
    def __init__(self, img_shape=(128, 128, 128), input_dim=4, output_dim=3,
                 embed_dim=768, patch_size=16, num_heads=12, dropout=0.0,
                 ext_layers=[3, 6, 9, 12], norm='instance',
                 base_filters=16,
                 dim_linear_block=3072):
        """
        Args:
            img_shape: volume shape, provided as a tuple
            input_dim: input modalities/channels
            output_dim: number of classes
            embed_dim: transformer embed dim.
            patch_size: the non-overlapping patches to be created
            num_heads: for the transformer encoder
            dropout: percentage for dropout
            ext_layers: transformer layers to use their output
            version: 'light' saves some parameters in the decoding part
            norm: batch or instance norm for the conv blocks
        """
        super().__init__()
        self.num_layers = 12
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d

        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim,
                                  cube_size=img_shape, patch_size=patch_size, dropout=dropout)

        self.transformer = TransformerEncoder(embed_dim, num_heads,
                                              self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)

        self.init_conv = Conv3DBlock(input_dim, base_filters, double=True, norm=self.norm)

        # blue blocks in Fig.1
        self.z3_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 2, layers=3)

        self.z6_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 4, layers=2)

        self.z9_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 8, layers=1)

        # Green blocks in Fig.1
        self.z12_deconv = TranspConv3DBlock(embed_dim, base_filters * 8)

        self.z9_deconv = TranspConv3DBlock(base_filters * 8, base_filters * 4)
        self.z6_deconv = TranspConv3DBlock(base_filters * 4, base_filters * 2)
        self.z3_deconv = TranspConv3DBlock(base_filters * 2, base_filters)

        # Yellow blocks in Fig.1
        self.z9_conv = Conv3DBlock(base_filters * 8 * 2, base_filters * 8, double=True, norm=self.norm)
        self.z6_conv = Conv3DBlock(base_filters * 4 * 2, base_filters * 4, double=True, norm=self.norm)
        self.z3_conv = Conv3DBlock(base_filters * 2 * 2, base_filters * 2, double=True, norm=self.norm)
        # out convolutions
        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(base_filters * 2, base_filters, double=True, norm=self.norm),
            # grey block, final classification layer
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1))

    def forward(self, x):
        transf_input = self.embed(x)
        z3, z6, z9, z12 = map(lambda t: rearrange(t, 'b (x y z) d -> b d x y z',
                                                  x=self.patch_dim[0], y=self.patch_dim[1], z=self.patch_dim[2]),
                              self.transformer(transf_input))

        # Blue convs
        z0 = self.init_conv(x)
        z3 = self.z3_blue_conv(z3)
        z6 = self.z6_blue_conv(z6)
        z9 = self.z9_blue_conv(z9)

        # Green block for z12
        z12 = self.z12_deconv(z12)
        # Concat + yellow conv
        y = torch.cat([z12, z9], dim=1)
        y = self.z9_conv(y)

        # Green block for z6
        y = self.z9_deconv(y)
        # Concat + yellow conv
        y = torch.cat([y, z6], dim=1)
        y = self.z6_conv(y)

        # Green block for z3
        y = self.z6_deconv(y)
        # Concat + yellow conv
        y = torch.cat([y, z3], dim=1)
        y = self.z3_conv(y)

        y = self.z3_deconv(y)
        y = torch.cat([y, z0], dim=1)
        return self.out_conv(y)
