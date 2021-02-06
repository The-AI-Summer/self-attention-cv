# Self-attention building blocks for computer vision applications in PyTorch

Implementation of self attention mechanisms for general purpose in PyTorch with einsum and einops.rearrange


Focused on computer vision self-attention modules. 

Ongoing repository. pip package coming soon...

## Background on attention and transformers
- [How Attention works in Deep Learning](https://theaisummer.com/attention/)
- [How Transformers work in deep learning and NLP](https://theaisummer.com/transformer/)
- How to implement multi-head self-attention blocks in PyTorch using the einsum notation


### Code Examples

```python
import torch
from self_attention import SelfAttentionAISummer

model = SelfAttentionAISummer(dim=64)
x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
mask = torch.zeros(10, 10)
mask[3:8, 3:8] = 1
y = model(x, mask)
```

```python
```



### Attention modules implemented so far:
- Scaled dot product self attention
- Multi-head-self-attention
- Axial attention and axial attention residual block

#### TODO
- Local attention for CV
- Botleneck self-attention 


