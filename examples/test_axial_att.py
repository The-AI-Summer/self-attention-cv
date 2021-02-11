import torch

from self_attention_cv import AxialAttention

in_channels = 256
dim = 32  # token's dim

model = AxialAttention(dim=32, in_channels=in_channels, heads=8)
a = torch.rand(4 * 64, in_channels, dim)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionAISummer OK')
