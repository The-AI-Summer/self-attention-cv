import torch

from self_attention_cv import ResNet50ViT
from self_attention_cv import ViT


def test_vit():
    model = ResNet50ViT(img_dim=128, pretrained_resnet=False, blocks=3, num_classes=10, dim_linear_block=256, dim=256)
    x = torch.rand(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 10)  # batch, classes
    print('ResNet50ViT inference complete')

    model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=10, dim=512)
    x = torch.rand(2, 3, 256, 256)
    y = model(x)  # [2,10]
    assert y.shape == (2, 10)  # batch, classes

    # The transformer can process any img dim that is a multiple of patch_dim
    x = torch.rand(2, 3, 160, 160)
    y = model(x)  # [2,10]
    assert y.shape == (2, 10)  # batch, classes
    print('ViT inference complete')
