import torch

from self_attention_cv import TransformerBlock, TransformerEncoder


def test_MultiHeadSelfAttention():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerEncoder(dim=64, blocks=6, heads=8).to(device)
    x = torch.rand(16, 10, 64).to(device)  # [batch, tokens, dim]
    mask = torch.zeros(10, 10).to(device) # tokens X tokens
    mask[5:8, 5:8] = 1
    mask = mask > 0 # convert to torch.cuda.BoolTensor or torch.BoolTensor
    y = model(x, mask)
    assert y.shape == x.shape
    print("Transformer AISummer OK")

    model = TransformerBlock(dim=64).to(device)
    x = torch.rand(16, 10, 64).to(device)  # [batch, tokens, dim]
    mask = torch.zeros(10, 10).to(device)  # tokens X tokens
    mask[5:8, 5:8] = 1
    mask = mask>0

    y = model(x, mask)
    assert y.shape == x.shape

    # forward without mask
    y = model(x)
    assert y.shape == x.shape
    print("Transformer block AISummer OK")


test_MultiHeadSelfAttention()