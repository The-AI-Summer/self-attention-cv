import torch
from self_attention import MultiHeadSelfAttentionAISummer

model = MultiHeadSelfAttentionAISummer(64)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
mask = torch.zeros(10, 64)  # tokens X tokens
mask[5:8, 5:8] = 1
y = model(x,mask)
assert y.shape == x.shape
print("MultiHeadSelfAttentionAISummer OK")