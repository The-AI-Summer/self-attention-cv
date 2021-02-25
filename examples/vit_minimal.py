import torch

from self_attention_cv import ViT

model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=10, dim=512)
x = torch.rand(2, 3, 256, 256)
y = model(x)  # [2,10]
assert y.shape == (2, 10)  # batch, classes

# The transformer can process any img dim that is a multiple of patch_dim
x = torch.rand(2, 3, 160, 160)
y = model(x)  # [2,10]
assert y.shape == (2, 10)  # batch, classes
print('ViT inference complete')
