import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, images):
        feats = self.base_model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats