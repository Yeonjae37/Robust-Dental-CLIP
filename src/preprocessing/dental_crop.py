import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import yaml
from PIL import Image
from ultralytics import YOLO

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

def load_cfg(path: Optional[str]):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_yolo(weights: str | Path) -> YOLO:
    return YOLO(str(weights))


def detect_teeth(model: YOLO, image: Image.Image, conf: float = 0.25) -> List[Box]:
    results = model.predict(source=image, conf=conf, verbose=False)
    res = results[0]
    boxes: List[Box] = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append((x1, y1, x2, y2))
    return boxes

def crop_by_boxes(image: Image.Image, boxes: List[Box]) -> List[Image.Image]:
    crops: List[Image.Image] = []
    w, h = image.width, image.height
    for (x1, y1, x2, y2) in boxes:
      x1 = max(0, int(x1))
      y1 = max(0, int(y1))
      x2 = min(w, int(x2))
      y2 = min(h, int(y2))
      crops.append(image.crop((x1, y1, x2, y2)))
    return crops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="yaml config", default=None)
    parser.add_argument("--image", help="input xray image path")
    parser.add_argument("--weights", help="yolo weights path")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save-dir", default="./crops")
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    image_path = args.image or cfg.get("paths", {}).get("image")
    weights = args.weights or cfg.get("paths", {}).get("weights")
    save_dir = args.save_dir or cfg.get("paths", {}).get("crops_dir", "./crops")
    conf = args.conf or cfg.get("detect", {}).get("conf", 0.25)

    img = Image.open(image_path).convert("RGB")
    model = load_yolo(weights)
    boxes = detect_teeth(model, img, conf=conf)
    print(f"detected {len(boxes)} teeth")

    if args.save_crops:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        for i, c in enumerate(crop_by_boxes(img, boxes)):
            out_path = save_dir / f"{stem}_tooth_{i:02d}.png"
            c.save(out_path)
            print(f"saved crop: {out_path}")

if __name__ == "__main__":
    main()