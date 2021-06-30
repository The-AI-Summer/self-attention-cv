"""
Example of weight loading of Timesformer from ViT trained from google on image classification
Inspired from https://github.com/m-bain/video-transformers/blob/c4fef59c1cc20d5454cff18ba88b309bd9d8a502/video-transformers/timesformer.py#L355
"""

from self_attention_cv import Timesformer

blocks = 12
dim_vit = 768
img_dim = 224
patch_dim = 16
in_channels = 3
mlp_ratio = 3

# Independent from the parameter loading
video_classes = 12
frames = 3

def show_layers_loaded(model_init, model):
    updated_layers = 0
    for current_params, loaded_params in zip(model_init.parameters(), model.parameters()):
        old_weight, new_weight = current_params.data, loaded_params.data
        if (old_weight - new_weight).sum() < 1e-6:
            updated_layers = updated_layers + 1

    print(f"Layers that have been loaded: {updated_layers}")


model = Timesformer(in_channels=in_channels, patch_dim=patch_dim, img_dim=img_dim, frames=frames, num_classes=video_classes,
                    blocks=blocks, dim=dim_vit,
                    dim_linear_block=mlp_ratio * dim_vit)

model_init = Timesformer(in_channels=in_channels, patch_dim=patch_dim, img_dim=img_dim, frames=frames, num_classes=video_classes,
                         blocks=blocks, dim=dim_vit,
                         dim_linear_block=mlp_ratio * dim_vit)

model_init.load_state_dict(model.state_dict())

# Option 1: need timm installed
import torch.nn as nn
from timm.models import vision_transformer

vit_model = vision_transformer.vit_base_patch16_224(pretrained=True)
vit_model.head = nn.Identity()
model.load_state_dict(vit_model.state_dict(), strict=False)

show_layers_loaded(model_init, model)

# Option 2: load from URL of timm without needing to install the timm libary
from torchvision.models.utils import load_state_dict_from_url

url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
model.load_state_dict(load_state_dict_from_url(url, progress=True), strict=False)
show_layers_loaded(model_init, model)


# Layers that have been loaded: 176