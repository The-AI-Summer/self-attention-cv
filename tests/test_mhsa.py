import torch
from self_attention import MultiHeadSelfAttentionAISummer

model = MultiHeadSelfAttentionAISummer(64)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
y = model(x)
assert y.shape == x.shape
print("MultiHeadSelfAttentionAISummer OK")