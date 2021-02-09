import torch

from self_attention_cv import AxialAttentionNaive
from self_attention_cv import AxialAttention

in_channels = 128
dim = 16

model = AxialAttention(in_channels=in_channels, dim=dim)
a = torch.rand(4 * 64, in_channels, dim)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionAISummer OK')

model = AxialAttentionNaive(in_channels=in_channels, dim=dim)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionNaiveAISummer OK')
