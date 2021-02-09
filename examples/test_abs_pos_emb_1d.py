import torch

from self_attention_cv.pos_embeddings import AbsPosEmb1D,RelPosEmb1D,rel_pos_emb_1d

model = AbsPosEmb1D(tokens=20, dim=64)
q = torch.rand(2, 3, 20, 64)
y = model(q)


print('abs:',y.shape)


model = RelPosEmb1D(tokens=20, dim=64)
q = torch.rand(2, 3, 20, 64)

y = rel_pos_emb_1d(q,torch.rand(39,64))


print('rel:',y.shape)

