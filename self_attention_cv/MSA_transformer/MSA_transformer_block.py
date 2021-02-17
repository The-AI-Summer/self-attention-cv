from einops import rearrange
from torch import nn

from .tied_axial_attention import TiedRowAxialAttention
from ..transformer_vanilla.mhsa import MultiHeadSelfAttention


class MSATransformerBlock(nn.Module):
    """
    MSA transformer block from the paper MSA Transformer
    Link: https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1
    """

    def __init__(self, *, dim, rows, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1, activation=nn.GELU):
        """
        Args:
            dim: token's vector length
            rows: number of rows with shared/tied attention that will be summed in Q*K^T
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
        """
        super().__init__()
        self.column_att = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.row_att = TiedRowAxialAttention(dim=dim, rows=rows, heads=heads, dim_head=dim_head)
        self.rows = rows
        self.dim = dim

        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        # as in vanilla transformer
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        assert x.dim() == 4, 'Ensure your input is 4D: [batch,channels, height,width]'
        # compose rows (h) with batch
        x = rearrange(x, 'b c h w -> (b h) c w')
        x = self.row_att(self.norm_1(x), mask) + x

        # decompose rows + merge batch with height
        x = rearrange(x, '(b h) c w  -> (b w) c h', h=self.rows)
        x = self.column_att(self.norm_2(x), mask) + x
        x = self.mlp(x) + x
        return rearrange(x, '(b w) c h -> b c h w', w=self.dim)


class MSATransformerEncoder(nn.Module):
    def __init__(self, *, dim, rows, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0):
        super().__init__()
        self.block_list = [MSATransformerBlock(dim=dim, rows=rows,
                                               heads=heads,
                                               dim_head=dim_head,
                                               dim_linear_block=dim_linear_block,
                                               dropout=dropout) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
