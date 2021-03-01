import torch

from self_attention_cv.Transformer3Dsegmentation import Transformer3dSeg

batch_size = 1
num_classes = 2
in_channels = 3
block = torch.rand(batch_size, 3, 24, 24, 24)
model = Transformer3dSeg(subvol_dim=24, patch_dim=8, in_channels=in_channels, blocks=2, num_classes=num_classes)
n = 24 // 8
y = model(block)
assert y.shape == (batch_size, num_classes, n, n, n)
print('forward pass Transformer3dSeg OK')
