import torch

from self_attention_cv import MSATransformerBlock, TiedRowAxialAttention

model = TiedRowAxialAttention(dim=128, rows=64)
x = torch.rand(2 * 64, 99, 128)
y = model(x)
assert x.shape == y.shape

model = MSATransformerBlock(dim=64, rows=64)
# batch channels h w
x = torch.rand(2, 40, 64, 64)
mask = torch.zeros(40, 40)
mask[15:35, 15:35] = 1
y = model(x, mask)
assert x.shape == y.shape
