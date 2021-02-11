from einops import rearrange
from torch import nn

from .axial_attention import AxialAttention
from ..transformer_vanilla.mhsa import MultiHeadSelfAttention


def _conv2d1x1(in_channels, out_channels, stride=1):
    """1x1 convolution for contraction and expansion of the channels dimension
    conv is followed by batch norm"""
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_channels))


class AxialAttentionBlock(nn.Module):
    def __init__(self, in_channels, dim, heads=8, axial_att=True, dim_head=None):
        """
        Axial-attention block implementation as described in:
        paper: https://arxiv.org/abs/2003.07853 , Fig. 2 page 7
        blogpost: TBA
        official code: https://github.com/csrhddlam/axial-deeplab
        Args:
            in_channels:
            dim: token's dim
            heads: the number of distict head representations
            axial_att: whether to use axial att or MHSA
            dim_head: for MHSA only
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        d_in = 128  # hardcoded

        # brings the input channels to 128 feature maps
        self.in_conv1x1 = _conv2d1x1(in_channels, d_in)
        self.out_conv1x1 = _conv2d1x1(d_in, in_channels)
        self.relu = nn.ReLU(inplace=True)

        if axial_att:
            self.dim_head = d_in // self.heads
            self.height_att = AxialAttention(dim=dim, in_channels=d_in, heads=heads)
            self.width_att = AxialAttention(dim=dim, in_channels=d_in, heads=heads)
        else:
            self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
            self.width_att = MultiHeadSelfAttention(dim=dim, dim_head=dim_head, heads=heads)
            self.height_att = MultiHeadSelfAttention(dim=dim, dim_head=dim_head, heads=heads)

    def forward(self, x_in):
        assert x_in.dim() == 4, 'Ensure your input is 4D: [batch,channels, height,width]'
        x = self.relu(self.in_conv1x1(x_in))
        # merge batch dim with width
        x = rearrange(x, 'b c h w -> (b w) c h')
        x = self.height_att(x)
        # decompose width + merge batch with height
        x = rearrange(x, '(b w) c h  -> (b h) c w', w=self.dim)
        x = self.relu(self.width_att(x))
        x = rearrange(x, '(b h) c w -> b c h w', h=self.dim)
        return self.relu(self.out_conv1x1(x) + x_in)
