import torch.nn as nn
from einops import rearrange

from .bottleneck_layer import Bottleneck
from .decoder import Up, SignleConv
from ..vit import ViT


class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=3072,
                 patch_size=8,
                 vit_transformer_dim=768,
                 vit_transformer=None,
                 vit_channels=None,
                 ):
        """
        My reimplementation of TransUnet based on the paper:
        https://arxiv.org/abs/2102.04306
        Badly written, many details missing and significantly differently
        from the authors official implementation (super messy code also :P ).
        My implementation doesnt match 100 the authors code.
        Basically I wanted to see the logic with vit and resnet backbone for
        shaping a unet model with long skip connections.

        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
            vit_transformer: pass your own version of vit
            vit_channels: the channels of your pretrained vit. default is 128*8
            patch_dim: for image patches of the vit
        """
        super().__init__()
        self.inplanes = 128
        self.patch_size = patch_size
        self.vit_transformer_dim = vit_transformer_dim
        vit_channels = self.inplanes * 8 if vit_channels is None else vit_channels

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16

        assert (self.img_dim_vit % patch_size == 0), "Vit patch_dim not divisible"

        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        # to project patches back - undoes vit's patchification
        token_dim = vit_channels * (patch_size ** 2)
        self.project_patches_back = nn.Linear(vit_transformer_dim, token_dim)
        # upsampling path
        self.vit_conv = SignleConv(in_ch=vit_channels, out_ch=512)
        self.dec1 = Up(vit_channels, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)
        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)
        x4 = self.conv1(x2)
        x8 = self.conv2(x4)
        x16 = self.conv3(x8)  # out shape of 1024, img_dim_vit, img_dim_vit
        y = self.vit(x16)  # out shape of number_of_patches, vit_transformer_dim

        # from [number_of_patches, vit_transformer_dim] -> [number_of_patches, token_dim]
        y = self.project_patches_back(y)

        # from [batch, number_of_patches, token_dim] -> [batch, channels, img_dim_vit, img_dim_vit]
        y = rearrange(y, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=self.img_dim_vit // self.patch_size, y=self.img_dim_vit // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size)

        y = self.vit_conv(y)
        y = self.dec1(y, x8)
        y = self.dec2(y, x4)
        y = self.dec3(y, x2)
        y = self.dec4(y)
        return self.conv1x1(y)
