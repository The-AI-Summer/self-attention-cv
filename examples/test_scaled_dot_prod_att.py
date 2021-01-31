import torch
from self_attention import SelfAttentionAISummer

model = SelfAttentionAISummer(64)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
mask = torch.zeros(10, 10)
mask[5:8, 5:8] = 1
y = model(x, mask)
assert x.shape == y.shape
print("SelfAttentionAISummer OK")
