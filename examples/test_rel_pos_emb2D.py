import torch

from self_attention_cv.pos_embeddings import RelPosEmb2D

dim = 32  # spatial dim of the feat map
feat_map_size = (dim, dim)
tokens = dim ** 2
dim_head = 128
batch = 2
heads = 4

model = RelPosEmb2D(
    feat_map_size=feat_map_size,
    dim_head=dim_head)

q = torch.rand(batch, heads, tokens, dim_head)
y = model(q)
assert y.shape == (batch, heads, tokens, tokens)
