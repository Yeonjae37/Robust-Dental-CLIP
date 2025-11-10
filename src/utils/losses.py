import torch
import torch.nn.functional as F

def clip_loss(logits_per_image, logits_per_text):
    bsz = logits_per_image.size(0)
    targets = torch.arange(bsz, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2