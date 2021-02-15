import torch

from self_attention_cv.bottleneck_transformer import BottleneckAttention, BottleneckModule

att = BottleneckAttention(dim=256, fmap_size=(32, 32), heads=4)
inp = torch.rand(1, 256, 32, 32)
y = att(inp)
assert y.shape == inp.shape

inp = torch.rand(1, 512, 32, 32)
# bottleneck_block = BottleneckBlock(in_channels=512, fmap_size=(32, 32), heads=4, out_channels=1024, pooling=True)
bottleneck_block = BottleneckModule(in_channels=512, fmap_size=(32, 32), heads=4, out_channels=1024, pooling=True)
y = bottleneck_block(inp)
print(y.shape)
