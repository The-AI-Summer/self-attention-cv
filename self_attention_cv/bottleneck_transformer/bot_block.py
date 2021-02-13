from torch import nn

from .bot_att import BottleneckAttention
from ..axial_attention_deeplab.axial_attention_residual_block import _conv2d1x1


class BottleneckBlock(nn.Module):
    def __init__(self, *, in_channels,
                 fmap_size,
                 out_channels=2048,
                 proj_factor=4,
                 heads=4,
                 dim_head=None,
                 pooling=False,
                 content_positional_embedding=True):
        """
        paper: https://arxiv.org/abs/2101.11605
        Check figure 3, and Table 1

        Args:
            in_channels: number of feat_maps
            fmap_size: spatial dims
            out_channels: the desired output channels of the layers
            proj_factor: used to calc. the bottleneck dim of the MHSA. (out_channels // proj_factor)
            heads: number of representation to learn from the input
            dim_head: defaults to (bottleneck_dimension/heads)
            pooling: (bool) whether to apply avg pool after bot-MHSA
            content_positional_embedding: (bool) whether to apply 2D rel pos enc
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
                                   expansion)  # no relu after expansion

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
        print('forward..', x.shape)
        return self.block(x) + self.shortcut(x)


class BottleneckModule(nn.Module):
    def __init__(self, *, in_channels,
                 fmap_size,
                 out_channels=2048,
                 proj_factor=4,
                 heads=4,
                 dim_head=None,
                 pooling=True,
                 content_positional_embedding=True,
                 num_layers=3,  # default
                 ):
        """
        Applies 3 bottleneck layers as in the original paper
        Args:
            in_channels: number of feat_maps
            fmap_size: spatial dims
            out_channels: the desired output channels of the layers
            proj_factor: used to calc. the bottleneck dim of the MHSA. (out_channels // proj_factor)
            heads: number of representation to learn from the input
            dim_head: defaults to (bottleneck_dimension/heads)
            pooling: (bool) whether to apply avg pool after bot-MHSA
            content_positional_embedding: (bool) whether to apply 2D rel pos enc
            num_layers: 3 used as in paper
        """

        super().__init__()
        block_list = []
        for i in range(num_layers):
            if i == 0:
                feat_map = fmap_size
                if pooling:
                    pool = True
            else:
                pool = False
                if pooling:
                    in_channels = out_channels
                    feat_map = (fmap_size[0] // 2, fmap_size[1] // 2)

            block_list.append(BottleneckBlock(in_channels=in_channels,
                                              fmap_size=feat_map,
                                              out_channels=out_channels,
                                              proj_factor=proj_factor,
                                              heads=heads,
                                              dim_head=dim_head,
                                              pooling=pool,
                                              content_positional_embedding=content_positional_embedding))
        self.model = nn.Sequential(*block_list)

    def forward(self, x):
        return self.model(x)
