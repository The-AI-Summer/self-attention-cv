import torch

from self_attention_cv import Timesformer

model = Timesformer(frames=3, img_dim=64, blocks=2, dim=512)
a = torch.rand(2, 3, 3, 64, 64)
b = model(a)
print(b.shape)
