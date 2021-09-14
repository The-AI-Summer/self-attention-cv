import torch

from self_attention_cv import TransUnet


def test_TransUnet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.rand(2, 3, 128, 128).to(device)
    model = TransUnet(in_channels=3, img_dim=128,
                      vit_blocks=1, classes=5, patch_size=4).to(device)
    y = model(a)
    print('final out shape:', y.shape)

test_TransUnet()
