import torch

from self_attention_cv import AxialAttentionBlock


def test_axial_att():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AxialAttentionBlock(in_channels=256, dim=64, heads=8).to(device)
    x = torch.rand(1, 256, 64, 64).to(device)  # [batch, tokens, dim, dim]
    y = model(x)
    assert y.shape == x.shape
    print('AxialAttentionBlockAISummer OK')
