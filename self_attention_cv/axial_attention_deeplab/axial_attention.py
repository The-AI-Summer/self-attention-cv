import torch
from einops import rearrange
from torch import nn

from self_attention_cv.pos_embeddings.relative_pos_enc_qkv import RelativePosEncQKV


def _conv1d1x1(in_channels, out_channels):
    """1D convolution with kernel size of 1 followed by batch norm"""
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                         nn.BatchNorm1d(out_channels))


class AxialAttention(nn.Module):
    def __init__(self, in_channels, dim, heads=8, dim_head_v=16, dim_head_kq=8):
        """
        Fig.1 page 6 in Axial DeepLab paper
        """
        super().__init__()
        self.dim = dim

        self.heads = heads
        self.heads_planes = in_channels // self.heads

        self.dim_head_v = dim_head_v
        self.dim_head_kq = dim_head_kq
        self.qkv_channels = self.dim_head_v + self.dim_head_kq * 2  #
        self.to_qvk = _conv1d1x1(in_channels, self.heads * self.qkv_channels)

        # Position embedding
        self.RelativePosEncQKV = RelativePosEncQKV(dim, self.heads_planes, self.dim_head_v, self.dim_head_kq)

        # Batch normalization - not common, but we dont need to scale down the dot products this way
        self.attention_norm = nn.BatchNorm2d(heads * 3)
        self.out_norm = nn.BatchNorm1d(in_channels * 2)

        assert self.heads_planes * 2 == self.qkv_channels, \
            f'2*in_channels// self.heads = {self.heads_planes * 2} must be equal to {self.qkv_channels}\
                     Set in_channels to 128 '

    def forward(self, x_in):
        assert x_in.dim() == 3, 'Ensure your input is 4D: [b * width, chan, height] or [b * height, chan, width]'

        # Calculate position embedding -> [ batch*width , qkv_channels,  dim ]
        qkv = self.to_qvk(x_in)

        qkv = rearrange(qkv, 'b (q h) d -> b h q d ', d=self.dim, q=self.qkv_channels, h=self.heads)

        # dim_head_kq != dim_head_v so I cannot decompose with einsum/einops here i think
        q, k, v = torch.split(qkv, [self.dim_head_kq, self.dim_head_kq, self.dim_head_v], dim=2)

        q_embedding, k_embedding, v_embedding = self.RelativePosEncQKV()

        # Computations are carried as Fig.1 page 6 in Axial DeepLab paper
        qr = torch.einsum('b h i d, i d j -> b h d j ', q, q_embedding)
        kr = torch.einsum('b h i d, i d j -> b h d j ', k, k_embedding)

        dots = torch.einsum('b h i d, b h i j -> b h d j', q, k)

        # We normalize the 3 tensors qr, kr, dots together before element-wise addition
        # To do so we concatenate the tensor heads just to normalize them
        # conceptually similar to scaled dot product in MHSA
        # Here n = len(list)
        norm_dots = self.attention_norm(rearrange(list([qr, kr, dots]), 'n b h d j -> b (h n) d j'))

        # Now we can decompose them
        norm_dots = rearrange(norm_dots, 'b (h n) d j -> n b h d j', n=3)

        # And use einsum in the n=3 axis for element-wise sum
        norm_dots = torch.einsum('n b h d j -> b h d j', norm_dots)

        # Last dimension is used softmax and matrix multplication
        attn = torch.softmax(norm_dots, dim=-1)
        # Matrix multiplication will be performed in the dimension of the softmax! Attention :)
        out = torch.einsum('b h d j,  b h i j -> b h i d', attn, v)

        # Last embedding of v
        kv = torch.einsum('b h d j, i d j -> b h i d ', attn, v_embedding)

        # To perform batch norm as described in paper,
        # we will merge the dimensions that are != self.dim
        # n = 2 = len(list)
        out = self.out_norm(rearrange(list([kv, out]), 'n b h i d ->  b (n h i ) d'))
        # decompose back output and merge heads
        out = rearrange(out, 'b (n h i ) d ->  n b (h i) d ', n=2, h=self.heads)
        # element wise sum in n=2 axis
        return torch.einsum('n b j i -> b j i', out)

