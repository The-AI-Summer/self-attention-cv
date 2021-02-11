from torch import nn

from .bot_att import BottleneckAttention
from ..axial_attention_deeplab.axial_attention_residual_block import _conv2d1x1


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels,
                 fmap_size,
                 out_channels=2048,
                 proj_factor=4,
                 heads=4,
                 dim_head=None,
                 pooling=False,
                 content_positional_embedding=True):
        """
        paper: https://arxiv.org/abs/2101.11605
        Figure 3, and Table 1
        """
        super().__init__()
        bottleneck_dimension = out_channels // proj_factor  # contraction_channels 512
        mhsa_out_channels = bottleneck_dimension if dim_head is None else dim_head * heads

        contraction = _conv2d1x1(in_channels, bottleneck_dimension)

        bot_mhsa = BottleneckAttention(
            dim=bottleneck_dimension,
            fmap_size=fmap_size,
            heads=heads,
            dim_head=dim_head,
            content_positional_embedding=content_positional_embedding)

        pool_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)) if pooling else nn.Identity()

        expansion = _conv2d1x1(mhsa_out_channels, out_channels)

        self.block = nn.Sequential(contraction,
                                   nn.ReLU(),
                                   bot_mhsa,
                                   pool_layer,
                                   nn.BatchNorm2d(mhsa_out_channels),
                                   nn.ReLU(),
                                   expansion) # no relu after expansion
        # TODO find init_zero=True tf param for batch norm

        # skip connection
        if pooling or in_channels != out_channels:
            # if we have pooling we need a 2-strided 1x1 conv
            stride = 2 if pooling else 1
            # we also need to match the desired out_channels from the Bot-MHSA
            self.shortcut = nn.Sequential(
                _conv2d1x1(in_channels, out_channels, stride=stride),
                nn.ReLU())
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
