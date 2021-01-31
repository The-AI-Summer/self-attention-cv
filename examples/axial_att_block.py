import torch

from self_attention import AxialAttentionBlockAISummer

model = AxialAttentionBlockAISummer(in_channels=32, dim=64)
x = torch.rand(4, 32, 64, 64)  # [batch, tokens, dim]

y = model(x)
assert y.shape == x.shape
print('AxialAttentionBlockAISummer OK')
