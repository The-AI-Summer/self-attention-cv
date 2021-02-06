import torch
from einops import rearrange
from torch import nn


def _conv1d1x1(in_channels, out_channels):
    """1D convolution with kernel size of 1 followed by batch norm"""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))


class AxialAttentionNaiveAISummer(nn.Module):
    def __init__(self, in_channels, dim, heads=8, dim_head_v=16, dim_head_kq=8):
        """
        Fig.1 page 6 in Axial DeepLab paper
        No batch normalization inside the MHSA layer
        Initial implementation for simplicity
        Instead I use the scale factor of dim ** -0.5 to q
        """
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.heads = heads
        self.heads_planes = in_channels // self.heads

        self.dim_head_v = dim_head_v
        self.dim_head_kq = dim_head_kq
        self.qkv_channels = self.dim_head_v + self.dim_head_kq * 2  #
        self.to_qvk = _conv1d1x1(in_channels, self.heads * self.qkv_channels)

        # Position embedding - position-sensitive self-attention
        self.relative = nn.Parameter(torch.randn(self.heads_planes * 2, dim * 2 - 1), requires_grad=True)

        assert self.heads_planes * 2 == self.qkv_channels, \
            f'2*in_channels// self.heads = {self.heads_planes * 2} must be equal to {self.qkv_channels}\
                     Set in_channels to 128 '

    def find_flattend_index(self):
        query_index = torch.arange(self.dim).unsqueeze(0)
        key_index = torch.arange(self.dim).unsqueeze(1)
        relative_index = key_index - query_index + self.dim - 1  # dim X dim
        return rearrange(relative_index, 'i j->(i j)')

    def forward(self, x_in):
        assert x_in.dim() == 3, 'Ensure your input is 4D: [b * width, chan, height] or [b * height, chan, width]'

        # Calculate position embedding -> [ batch*width , qkv_channels,  dim ]
        qkv = self.to_qvk(x_in)

        qkv = rearrange(qkv, 'b (q h) d -> b h q d ', d=self.dim, q=self.qkv_channels, h=self.heads)

        # dim_head_kq != dim_head_v so I cannot decompose with einsum/einops here i think
        q, k, v = torch.split(qkv, [self.dim_head_kq, self.dim_head_kq, self.dim_head_v], dim=2)

        flatten_index = self.find_flattend_index()

        all_embeddings = torch.index_select(self.relative, 1, flatten_index)
        all_embeddings = rearrange(all_embeddings, 'c (x y) -> c x y', x=self.dim)

        # resulting shape [dim_head, dim, dim] , shared across heads
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.dim_head_kq, self.dim_head_kq, self.dim_head_v],
                                                            dim=0)
        # Instead of batch norm inside the layer
        q = q * self.scale

        # Computations are carried as Fig.1 page 6 in Axial DeepLab paper
        qr = torch.einsum('b h i d, i d j -> b h d j ', q, q_embedding)
        kr = torch.einsum('b h i d, i d j -> b h d j ', k, k_embedding)

        dots = torch.einsum('b h i d, b h i j -> b h d j', q, k)

        # Last dimension is used softmax and matrix multplication
        attn = torch.softmax(qr + kr + dots, dim=-1)
        # Matrix multiplication will be performed in the dimension of the softmax! Attention :)
        out = torch.einsum('b h d j,  b h i j -> b h i d', attn, v)

        # Last embedding of v
        kv = torch.einsum('b h d j, i d j -> b h i d ', attn, v_embedding)

        return rearrange(kv + out, "b h i d -> b (h i) d")
