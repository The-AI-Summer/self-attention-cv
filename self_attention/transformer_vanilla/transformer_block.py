from torch import nn

from .mhsa import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x, mask=None):
        y = self.norm(self.mhsa(x, mask)) + x
        return self.linear(y) + y


class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
