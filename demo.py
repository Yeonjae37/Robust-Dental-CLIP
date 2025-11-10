import argparse, json
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

TEXTS = [
    "a dental x-ray of a tooth with impacted condition",
    "a dental x-ray of a tooth with caries",
    "a dental x-ray of a tooth with periapical lesion",
    "a dental x-ray of a tooth with deep caries",
    "a dental x-ray of a healthy tooth",
]
ID2NAME = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries", "Healthy"]

LABEL_COLORS = {
    "Impacted": (255, 0, 0),             # 빨강
    "Caries": (0, 255, 0),               # 초록
    "Periapical Lesion": (0, 128, 255),  # 파랑
    "Deep Caries": (255, 165, 0),        # 주황
    "Healthy": (255, 255, 0),            # 노랑
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--out-img", default="demo_out.png")
    p.add_argument("--out-json", default="demo_out.json")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.yolo_weights)
    clip, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip = clip.to(device).eval()

    txt_tokens = tokenizer(TEXTS).to(device)
    with torch.no_grad():
        txt_feats = clip.encode_text(txt_tokens)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    img_path = Path(args.image)
    orig_img = Image.open(img_path).convert("RGB")
    W, H = orig_img.size

    legend_w = 220
    canvas = Image.new("RGB", (W + legend_w, H), (255, 255, 255))
    canvas.paste(orig_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    results = yolo.predict(source=orig_img, conf=args.conf, verbose=False)[0]

    detections = []

    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        crop = orig_img.crop((int(x1), int(y1), int(x2), int(y2)))
        im_t = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            im_feat = clip.encode_image(im_t)
            im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)
            logits = (im_feat @ txt_feats.T)
            probs = logits.softmax(dim=-1)[0].cpu().tolist()

        pred_idx = int(torch.tensor(probs).argmax().item())
        label = ID2NAME[pred_idx]
        color = LABEL_COLORS.get(label, (255, 0, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_pos = (x1 + 3, y1 + 3)
        draw.text(text_pos, label, fill=color)

        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "label": label,
                "probs": probs,
            }
        )

    legend_x = W + 10
    legend_y = 10
    line_h = 24
    draw.text((legend_x, legend_y), "Legend", fill=(0, 0, 0))
    legend_y += line_h
    for label in ID2NAME:
        c = LABEL_COLORS[label]

        draw.rectangle(
            [legend_x, legend_y, legend_x + 20, legend_y + 20],
            fill=c,
            outline=(0, 0, 0),
            width=1,
        )

        draw.text((legend_x + 26, legend_y), label, fill=(0, 0, 0))
        legend_y += line_h

    canvas.save(args.out_img)
    with open(args.out_json, "w") as f:
        json.dump({"image": img_path.name, "detections": detections}, f, indent=2)

if __name__ == "__main__":
    main()