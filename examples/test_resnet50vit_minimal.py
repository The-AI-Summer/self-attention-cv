import torch

from self_attention_cv import ResNet50ViT

model = ResNet50ViT(img_dim=128, pretrained_resnet=False, blocks=3, num_classes=10, dim_linear_block=256, dim=256)
x = torch.rand(2, 3, 128, 128)
y = model(x)
assert y.shape == (2, 10)  # batch, classes
print('ResNet50ViT inference complete')
