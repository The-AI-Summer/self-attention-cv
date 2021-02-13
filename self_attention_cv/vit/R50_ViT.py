import torch.nn as nn
from torchvision.models import resnet50

from .vit import ViT


class ResNet50ViT(nn.Module):
    def __init__(self, *, img_dim, pretrained_resnet=False,
                 resnet_layers=5,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=64,
                 dropout=0, transformer=None, classification=True
                 ):
        """
        ResNet50 + ViT for image classification
        Args:
            img_dim: the spatial image size
            pretrained_resnet: wheter to load pretrained weight from torch vision
            resnet_layers: use 5 or 6. the layer to keep from the resnet 50 backbone
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
        """
        super().__init__()
        assert 5 <= resnet_layers <= 6, f'Inser 5 or 6 resnet layers to keep'
        resnet_channels = 256 if resnet_layers == 5 else 512
        feat_dim = img_dim // 4 if resnet_layers == 5 else img_dim // 8

        resnet_layers = list(resnet50(pretrained=pretrained_resnet).children())[:resnet_layers]
        self.img_dim = img_dim

        res50 = nn.Sequential(*resnet_layers)

        vit = ViT(img_dim=feat_dim, in_channels=resnet_channels, patch_dim=1,
                  num_classes=num_classes, dim_linear_block=dim_linear_block, dim=dim,
                  dim_head=dim_head, dropout=dropout, transformer=transformer,
                  classification=classification, heads=heads, blocks=blocks)
        self.model = nn.Sequential(res50, vit)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == 3, f'Insert input with 3 inp channels, {c} received.'
        assert self.img_dim == h == w, f'Insert input with {self.img_dim} dimensions.'
        return self.model(x)
