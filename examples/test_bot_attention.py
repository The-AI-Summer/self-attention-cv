import torch
from self_attention_cv.bottleneck_transformer import BottleneckAttention

att = BottleneckAttention(dim=256, fmap_size=(32, 32), heads=4)
inp = torch.rand(1, 256, 32, 32)
y = att(inp)
print(y.shape)
