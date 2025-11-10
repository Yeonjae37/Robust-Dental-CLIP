import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, text_tokens):
        feats = self.base_model.encode_text(text_tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats