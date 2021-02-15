import torch.nn as nn
from einops import rearrange

from .bottleneck_layer import Bottleneck
from .decoder import Up, SignleConv
from ..vit import ViT


class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=4,
                 vit_dim_linear_mhsa_block=1024,
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
        """
        super().__init__()
        self.inplanes = 128
        vit_channels = self.inplanes * 8

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16
        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # encoder channels
                       patch_dim=1,
                       dim=vit_channels,  # vit out channels for decoding
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False)

        self.vit_conv = SignleConv(in_ch=vit_channels, out_ch=512)

        self.dec1 = Up(1024, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)
        self.conv1x1 = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)  # 128,64,64
        x4 = self.conv1(x2)  # 256,32,32
        x8 = self.conv2(x4)  # 512,16,16
        x16 = self.conv3(x8)  # 1024,8,8
        y = self.vit(x16)
        y = rearrange(y, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)
        y = self.vit_conv(y)
        y = self.dec1(y, x8)  # 256,16,16
        y = self.dec2(y, x4)
        y = self.dec3(y, x2)
        y = self.dec4(y)
        return self.conv1x1(y)
