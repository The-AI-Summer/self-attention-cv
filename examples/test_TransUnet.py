from self_attention_cv.transunet import TransUnet
import torch

a = torch.rand(2,3,128,128)

model = TransUnet(in_channels=3,img_dim=128)
y = model(a)
print(y.shape)
