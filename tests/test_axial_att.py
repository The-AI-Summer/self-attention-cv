import torch

from self_attention import AxialAttentionAISummer

model = AxialAttentionAISummer(in_channels=128, dim=64)
a = torch.rand(4 * 64, 128, 64)
y = model(a)
assert y.shape == a.shape
print('AxialAttentionAISummer OK')
