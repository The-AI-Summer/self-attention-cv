import torch
import torch.nn as nn
from einops import rearrange
from .bottleneck_layer import Bottleneck
from ..vit import ViT


class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels,
                 vit_blocks=1,
                 vit_heads=4,
                 vit_dim_linear_mhsa_block=512,
                 ):
        super().__init__()
        self.inplanes = 64

        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv1 = Bottleneck(self.inplanes, 128)
        self.conv2 = Bottleneck(128, 256, stride=2)
        self.conv3 = Bottleneck(256, 512, stride=2)

        self.img_dim = img_dim//16
        self.vit = ViT(img_dim=self.img_dim,
                       in_channels=512, # based on resnet channels
                       patch_dim=1,
                       dim=512, # out channels for decoding
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False)

    def forward(self, x):
        # ResNet 50 encoder
        x1 = self.init_conv(x)
        x2 = self.pool(x1)
        x2 = self.conv1(x2)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        # Vision Transformer ViT
        x6 = self.vit(x4)
        x7 = rearrange(x6, ' b (x y) dim -> b dim x y ', x=self.img_dim, y=self.img_dim)
        
        # Decoder

        return x7
