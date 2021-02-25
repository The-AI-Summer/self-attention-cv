import torch

from self_attention_cv import TransformerBlock

model = TransformerBlock(dim=64)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
mask = torch.zeros(10, 10)  # tokens X tokens
mask[5:8, 5:8] = 1
y = model(x, mask)
assert y.shape == x.shape
print("Transformer block AISummer OK")
