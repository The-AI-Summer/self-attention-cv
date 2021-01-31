import torch

from found.self_att_einsum import SelfAttentionEinSum
from found.self_att import SelfAttention


input_tensor = torch.rand(10,128,256)

self_att_einsum = SelfAttentionEinSum(dim=256, heads=4)
self_att  = SelfAttention( k=256, heads=4)

res1 = self_att_einsum(input_tensor)
res2 = self_att(input_tensor)
assert (res1.shape) == (res2.shape)