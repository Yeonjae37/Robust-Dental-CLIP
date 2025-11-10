import argparse
from pathlib import Path
import csv

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

TEXTS = [
    "a dental x-ray of a tooth with impacted condition",
    "a dental x-ray of a tooth with caries",
    "a dental x-ray of a tooth with periapical lesion",
    #"a dental x-ray of a healthy tooth",
]

LABEL2IDX = {
    "Impacted": 0,
    "Caries": 1,
    "Periapical Lesion": 2,
    # "Healthy": 3,
}

class CsvImageDataset(Dataset):
    def __init__(self, images_dir: str, labels_csv: str, preprocess):
        self.images_dir = Path(images_dir)
        self.preprocess = preprocess
        self.items = []

        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["image"]
                lbl = row["label"]
                if lbl not in LABEL2IDX:
                    raise ValueError(f"unknown label in csv: {lbl}")
                label_id = LABEL2IDX[lbl]
                self.items.append((fname, label_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label_id = self.items[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        return img, label_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="folder with images")
    parser.add_argument("--labels-csv", required=True, help="CSV with columns: image,label")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="checkpoints/clip_finetuned.pt")
    parser.add_argument("--model", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # base CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    dataset = CsvImageDataset(args.images_dir, args.labels_csv, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    tokenizer = open_clip.get_tokenizer(args.model)
    text_tokens = tokenizer(TEXTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)  # (5, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ text_features.T) * model.logit_scale.exp()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

        print(
            f"[epoch {epoch+1}/{args.epochs}] "
            f"loss={total_loss/total:.4f} acc={correct/total:.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "texts": TEXTS,
            "label2idx": LABEL2IDX,
            "clip_model": args.model,
            "clip_pretrained": args.pretrained,
        },
        out_path,
    )

if __name__ == "__main__":
    main()