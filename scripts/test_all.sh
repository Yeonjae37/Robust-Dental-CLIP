set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

TEST_IMG_DIR="$ROOT_DIR/data/cropped/test/images"

TEST_GT_CSV="$ROOT_DIR/data/cropped/test/labels.csv"

PRED_CSV="$ROOT_DIR/data/cropped/test/preds.csv"

python3 "$ROOT_DIR/src/inference/dental_inference_dir.py" \
  --crops-dir "$TEST_IMG_DIR" \
  --out-csv "$PRED_CSV"

python3 "$ROOT_DIR/src/eval/eval.py" \
  --gt-csv "$TEST_GT_CSV" \
  --pred-csv "$PRED_CSV"