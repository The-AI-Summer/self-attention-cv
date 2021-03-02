import torch

from self_attention_cv import Timesformer, SpacetimeMHSA

model_time = SpacetimeMHSA(dim=64, tokens_to_attend=10, space_att=False)
model_space = SpacetimeMHSA(dim=64, tokens_to_attend=128, space_att=True, linear_spatial_attention=True, k=128)
input_seq = torch.rand(2, 10 * 128 + 1, 64)
y = model_time(input_seq)
print('SpacetimeMHSA time ok')
y = model_space(y)
print('SpacetimeMHSA space ok')
print(y.shape)

model = Timesformer(img_dim=64, frames=4, num_classes=10, blocks=2, dim=512, linear_spatial_attention=True, k=128)
a = torch.rand(2, 4, 3, 64, 64)
b = model(a)
print(b.shape)
assert b.shape == (2, 10)
print('Video classifiction Timesformer ok')
