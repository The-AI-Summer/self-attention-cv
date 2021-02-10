import torch

from self_attention_cv.pos_embeddings import AbsPosEmb1D,RelPosEmb1D

model = AbsPosEmb1D(tokens=20, dim_head=64)
# batch heads tokens dim_head
q = torch.rand(2, 3, 20, 64)
y1 = model(q)

model = RelPosEmb1D(tokens=20, dim_head=64, heads=3)
q = torch.rand(2, 3, 20, 64)
y2 = model(q)

assert y2.shape == y1.shape
print('abs and pos emb ok')


