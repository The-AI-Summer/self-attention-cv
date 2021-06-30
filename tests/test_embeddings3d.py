from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
import torch


def test_emb_3d():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Embeddings3D(input_dim=1,embed_dim=768,cube_size=(128,128,128),patch_size=16,dropout=0.1).to(device)
    patches = int((128**3) / (16**3))
    a = torch.rand(1,1,128,128,128).to(device)

    b = model(a)
    assert b.shape == (1,patches,768)
