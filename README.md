# Robust Dental-CLIP via Noise Curriculum Learning

This repo contains a **CLIP-based pipeline** for tooth-level disease classification on dental panoramic X-rays.
To make CLIP more robust to this, we plan to apply noise-aware curriculum learning. 

1. **Detect & crop teeth** from a full X-ray (YOLO).
2. **classify each tooth** with CLIP into 5 labels  
   - Impacted
   - Caries
   - Periapical Lesion
   - Deep Caries
   - Healthy

### Environment
We use uv venv and the repo already has `pyproject.toml`, `uv.lock`
1. Create uv venv `uv venv`

2. Install required packages `uv sync`

3. Activate uv venv `source .venv/bin/activate`

### Datasets
We use the **Dentex** dataset:
- Link: https://www.kaggle.com/datasets/truthisneverlinear/dentex-challenge-2023
- Description: Dentex is a panoramic dental X-ray dataset that provides **image-level X-ray files** and **COCO-style annotations** for each tooth. Each annotation includes:

### Pretrained YOLO tooth detector

We trained a YOLOv8-based detector to localize individual teeth on panoramic dental X-rays from DENTEX.  

- Download the trained weights here: **[YOLO teeth weights (Google Drive)](https://drive.google.com/file/d/1yqhbHIsGE0P2jRrrJrbbQ1gkAzBuL4au/view?usp=drive_link)**
- Save it to: `weights/yolo_teeth_best.pt`

### YOLO Tooth Cropping
Detect teeth on a full X-ray and save each tooth crop.
```
python src/preprocessing/dental_crop.py \
  --image x-ray.png \
  --weights weights/yolo_teeth_best.pt \
  --save-crops \
  --save-dir ./crops
```

### Dental Inference (single image)
Classify one cropped tooth into 5 categories:
```
python -m src.inference.dental_inference \
  --image teeth.png
```