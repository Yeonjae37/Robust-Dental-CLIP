import torch
import torch.nn as nn

class CLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, logit_scale):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale

    def forward(self, images, text_tokens):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_tokens)
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text