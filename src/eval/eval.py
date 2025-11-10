import argparse
from pathlib import Path
import csv

def load_csv_flex(path: Path) -> dict[str, str]:

    text = path.read_text().strip()
    if not text:
        return {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.lower() for c in reader.fieldnames]

        fname_key = None
        for cand in ("filename", "image"):
            if cand in fieldnames:
                fname_key = cand
                break

        label_key = None
        for cand in ("label", "gt_label", "pred_label"):
            if cand in fieldnames:
                label_key = cand
                break

        out: dict[str, str] = {}
        for row in reader:
            fname = row[fname_key].strip()
            label = row[label_key].strip()
            out[fname] = label
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--pred-csv", required=True)
    args = parser.parse_args()

    gt = load_csv_flex(Path(args.gt_csv))
    pred = load_csv_flex(Path(args.pred_csv))

    total = 0
    correct = 0

    for fname, gt_label in gt.items():
        total += 1
        pred_label = pred.get(fname)
        if pred_label is not None and pred_label == gt_label:
            correct += 1

    print("===== EVAL RESULT =====")
    print(f"GT samples: {total}")
    print(f"Correct   : {correct}")
    if total:
        print(f"Acc       : {correct/total:.4f}")
    else:
        print("Acc       : N/A (no samples)")


if __name__ == "__main__":
    main()