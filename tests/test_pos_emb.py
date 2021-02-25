import torch

from self_attention_cv.pos_embeddings import AbsPosEmb1D, RelPosEmb1D, RelPosEmb2D


def test_pos_emb():
    model = AbsPosEmb1D(tokens=20, dim_head=64)
    # batch heads tokens dim_head
    q = torch.rand(2, 3, 20, 64)
    y1 = model(q)

    model = RelPosEmb1D(tokens=20, dim_head=64, heads=3)
    q = torch.rand(2, 3, 20, 64)
    y2 = model(q)

    assert y2.shape == y1.shape
    print('abs and pos emb ok')

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
