import torch
from self_attention_cv import UNETR


def test_unettr():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNETR(img_shape=(64, 64, 64), input_dim=1, output_dim=1, version='a').to(device)
    a = torch.rand(1, 1, 64, 64, 64).to(device)
    assert model(a).shape == (1,1,64,64,64)
    del model
    model = UNETR(img_shape=(64, 64, 64), input_dim=1, output_dim=1, version='light').to(device)
    assert model(a).shape == (1, 1, 64, 64, 64)

test_unettr()