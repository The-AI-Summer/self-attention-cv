import torch

from self_attention_cv import LinformerAttention, LinformerEncoder


def test_linformer():
    model = LinformerAttention(dim=64, shared_projection=False, proj_shape=(512, 128))
    a = torch.rand(1, 512, 64)
    y = model(a)
    assert y.shape == a.shape

    model = LinformerEncoder(dim=64, tokens=512)
    y = model(a)
    assert y.shape == a.shape
