import argparse
from pathlib import Path
import torch
import open_clip
from PIL import Image

CLIP_TEXTS = [
    "a dental x-ray of a tooth with impacted condition",
    "a dental x-ray of a tooth with caries",
    "a dental x-ray of a tooth with periapical lesion",
    "a dental x-ray of a tooth with deep caries",
    "a dental x-ray of a healthy tooth",
]

ID2NAME = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
    4: "Healthy",
}

def load_clip(device: str):
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    model = model.to(device)
    model.eval()

    text_tokens = tokenizer(CLIP_TEXTS).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return model, preprocess, text_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="cropped tooth image path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, text_feats = load_clip(device)

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = (img_feat @ text_feats.T) * model.logit_scale.exp()
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    pred_name = ID2NAME[pred_id]

    prob_str = ", ".join(f"{ID2NAME[i]}={probs[i]:.3f}" for i in range(len(ID2NAME)))
    print(f"image: {img_path.name}")
    print(f"pred: {pred_name}")
    print(f"probs: {prob_str}")


if __name__ == "__main__":
    main()