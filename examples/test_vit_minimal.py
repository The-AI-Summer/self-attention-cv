import torch

from self_attention_cv import ViT

model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=10)
x = torch.rand(2, 3, 256, 256)
y = model(x)
assert y.shape == (2, 10)  # batch, classes
print('ViT inference complete')
