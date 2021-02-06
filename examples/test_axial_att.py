import torch

from self_attention import AxialAttentionNaiveAISummer
from self_attention import AxialAttentionAISummer

in_channels = 128
dim = 16

model = AxialAttentionAISummer(in_channels=in_channels, dim=dim)
a = torch.rand(4 * 64, in_channels, dim)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionAISummer OK')

model = AxialAttentionNaiveAISummer(in_channels=in_channels, dim=dim)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionNaiveAISummer OK')
