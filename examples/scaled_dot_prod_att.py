import torch

from self_attention_cv import SelfAttention

model = SelfAttention(dim=4)
x = torch.rand(1, 3, 4)  # [batch, tokens, dim]
y = model(x)
assert x.shape == y.shape
print("SelfAttentionAISummer OK")
