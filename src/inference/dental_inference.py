import argparse
from pathlib import Path
import torch
import open_clip
from PIL import Image

TEXTS = [
    "a dental x-ray of a tooth with impacted condition",
    "a dental x-ray of a tooth with caries",
    "a dental x-ray of a tooth with periapical lesion",
    #"a dental x-ray of a healthy tooth",
]

ID2NAME = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    #3: "Healthy",
}

def load_clip(device: str, model_name: str = "ViT-B-16", pretrained: str = "openai", ckpt: str | None = None):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device).eval()

    if ckpt is not None:
        ckpt_path = Path(ckpt)
        data = torch.load(ckpt_path, map_location=device)
        state_dict = data.get("model", data)
        model.load_state_dict(state_dict, strict=False)

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(TEXTS).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return model, preprocess, text_feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="cropped tooth image path")
    parser.add_argument("--ckpt", default=None, help="finetuned CLIP checkpoint (.pt)")
    parser.add_argument("--clip-model", default="ViT-B-16")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, text_feats = load_clip(
        device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        ckpt=args.ckpt,
    )

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