import argparse
from pathlib import Path

import torch
import open_clip
from PIL import Image

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.clip import CLIPMode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="fine-tuned CLIP checkpoint (.pt or .pth)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base CLIP from open_clip
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    base_model = base_model.to(device).eval()

    # wrappers
    image_encoder = ImageEncoder(base_model).to(device)
    text_encoder = TextEncoder(base_model).to(device)
    model = CLIPModel(image_encoder, text_encoder, base_model.logit_scale).to(device).eval()

    # optional checkpoint
    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

    img = Image.open("test.png").convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a dental x-ray",
    ]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(img, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    for t, p in zip(texts, probs):
        print(f"{t}: {float(p):.4f}")

if __name__ == "__main__":
    main()