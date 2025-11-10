import argparse
from pathlib import Path
import csv
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

def load_clip(device: str, model_name: str = "ViT-B-16", pretrained: str = "openai", ckpt: str | None = None,):
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

        texts = TEXTS
    else:
        texts = TEXTS

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    return model, preprocess, text_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with cropped tooth images")
    ap.add_argument("--out-csv", required=True, help="where to save predictions")
    ap.add_argument("--ext", default="png")
    ap.add_argument("--ckpt", default=None, help="finetuned CLIP checkpoint (.pt)")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--clip-pretrained", default="openai")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, text_feats = load_clip(
        device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        ckpt=args.ckpt,
    )

    rows = []

    for img_path in sorted(data_dir.glob(f"*.{args.ext}")):
        img = Image.open(img_path).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ text_feats.T) * model.logit_scale.exp()
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        pred_label = ID2NAME[pred_id]

        print(f"{img_path.name}: {pred_label}")
        rows.append((img_path.name, pred_label))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred_label"])
        writer.writerows(rows)

if __name__ == "__main__":
    main()