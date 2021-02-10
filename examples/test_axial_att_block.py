import torch

from self_attention_cv import AxialAttentionBlock

model = AxialAttentionBlock(in_channels=256, dim=64)
x = torch.rand(1, 256, 64, 64)  # [batch, tokens, dim, dim]

y = model(x)
assert y.shape == x.shape
print('AxialAttentionBlockAISummer OK')
