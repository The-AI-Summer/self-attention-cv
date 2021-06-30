import torch

from self_attention_cv.bottleneck_transformer import BottleneckAttention, BottleneckModule


def test_bot_att():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    att = BottleneckAttention(dim=256, fmap_size=(32, 32), heads=4).to(device)
    inp = torch.rand(1, 256, 32, 32).to(device)
    y = att(inp)
    assert y.shape == inp.shape

    inp = torch.rand(1, 512, 32, 32).to(device)
    bottleneck_block = BottleneckModule(in_channels=512, fmap_size=(32, 32), heads=4, out_channels=1024, pooling=True).to(device)
    y = bottleneck_block(inp)
