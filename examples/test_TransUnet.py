import torch

from self_attention_cv.transunet import TransUnet

a = torch.rand(2, 3, 128, 128)

model = TransUnet(in_channels=3, img_dim=128, classes=5)
y = model(a)
print('final out shape:', y.shape)
