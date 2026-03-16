# 🧪 TRAINING_GUIDE.md — Blind-Project Complete Training Guide

Full walkthrough for training the Telugu campus navigation system from scratch.
All computation is LOCAL — no cloud services required.

---

## 1. Overall Pipeline

```
Step 1  →  Install dependencies
Step 2  →  Auto-download datasets (VizWiz + COCO + IndicCaption Telugu)
Step 3  →  Manual download: campus datasets (Kaggle + Roboflow)
Step 4  →  Train YOLOv11s detector  (COCO + campus combined)
Step 5  →  Train BLIP captioner     (Telugu-primary: 60% Telugu)
Step 6  →  Evaluate captioner
Step 7  →  Export models
Step 8  →  Enable in config.py
Step 9  →  Run live application
```

---

## 2. Dataset Details

### 2A. Object Detection Datasets (YOLO)

These teach the model to **SEE and locate** objects:

| Dataset | Size | Classes | How to Get |
|---------|------|---------|------------|
| MS-COCO 2017 | ~20 GB | 80 general objects | `python data/download_datasets.py --dataset coco` |
| Indoor Objects | ~1 GB | door, chair, table, window, pole | Kaggle — see MANUAL_DOWNLOADS.md |
| Door+Stairs+Chairs | ~500 MB | door, stairs, chair, toilet | Roboflow — see MANUAL_DOWNLOADS.md |
| SmartCane Indoor | ~300 MB | chair, table, door | Roboflow — see MANUAL_DOWNLOADS.md |

👉 **Manual downloads:** [`data/MANUAL_DOWNLOADS.md`](../data/MANUAL_DOWNLOADS.md)

### 2B. Caption Datasets (BLIP — Telugu)

These teach the model to **describe scenes in Telugu**:

| Dataset | Images | Language | Weight | How to Get |
|---------|--------|----------|--------|------------|
| **AI4Bharat IndicCaption** | ~40,000 | **Telugu** | **60%** | `python data/download_datasets.py --dataset indic` |
| VizWiz-Captions | 23,431 | English | 25% | `python data/download_datasets.py --dataset vizwiz` |
| MS-COCO Captions | 118,000 | English | 15% | `python data/download_datasets.py --dataset coco` |

#### Why Telugu Gets 60% Weight
The app must output **natively accurate Telugu**. By giving Telugu IndicCaption
60% of training batches, BLIP learns to generate Telugu text directly without
needing a translation step — faster, more natural, more accurate.

---

## 3. Step-by-Step Training

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
pip install datasets huggingface-hub     # For IndicCaption download
pip install deep-translator edge-tts pygame   # For Telugu TTS
```

### Step 2 — Auto-Download Datasets

```bash
# Download all auto-downloadable datasets
python data/download_datasets.py --dataset all

# Or individually:
python data/download_datasets.py --dataset indic    # Telugu captions (fastest)
python data/download_datasets.py --dataset vizwiz   # Blind user photos
python data/download_datasets.py --dataset coco     # COCO (~20 GB, takes time)
```

### Step 3 — Manual Campus Datasets

```bash
# See instructions:
python data/download_datasets.py --dataset manual-info
# OR open: data/MANUAL_DOWNLOADS.md
```

Place each dataset in `data/indoor_campus/<subfolder>/`.

### Step 4 — Train YOLO Detector

```bash
# Recommended: combined COCO + indoor campus (best for college premises)
python training/train_detector.py --dataset combined --model yolo11s.pt

# COCO only (if campus datasets not yet downloaded)
python training/train_detector.py --dataset coco --model yolo11s.pt

# Low VRAM (4 GB):
python training/train_detector.py --dataset combined --model yolo11s.pt --batch-size 8
```

| Model | VRAM | Accuracy | Recommended |
|-------|------|----------|-------------|
| `yolo11n.pt` | ~2 GB | ⭐⭐⭐ | Demo only |
| `yolo11s.pt` | ~3 GB | ⭐⭐⭐⭐ | **✅ Final app** |
| `yolo11m.pt` | ~5 GB | ⭐⭐⭐⭐⭐ | If you have RTX 4060+ |

Output: `checkpoints/yolo11_custom.pt`

### Step 5 — Train BLIP Captioner (Telugu)

```bash
# Default (8 epochs, Telugu-primary)
python training/train_captioner.py

# Low VRAM:
python training/train_captioner.py --batch-size 2

# More accuracy:
python training/train_captioner.py --epochs 12

# Resume from checkpoint:
python training/train_captioner.py --resume checkpoints/blip_finetuned/checkpoint_epoch4
```

Memory optimisations (for RTX 3050 / 4 GB):
- Gradient checkpointing → −40% VRAM
- Mixed precision (FP16)
- Gradient accumulation ×4 → effective batch = 16

Output: `checkpoints/blip_finetuned/best/`

### Step 6 — Evaluate

```bash
python training/evaluate.py
```

Target metrics on VizWiz validation set:

| Metric | Target | Meaning |
|--------|--------|---------|
| BLEU-1 | ≥ 0.55 | Word-level match (Telugu scripts) |
| BLEU-4 | ≥ 0.25 | Sentence quality |
| METEOR | ≥ 0.28 | Synonym-aware match |

### Step 7 — Export (Optional — for deployment)

```bash
python training/export_models.py
```

Exports ONNX and OpenVINO IR formats for CPU/edge deployment.

### Step 8 — Enable Trained Models

Open `config.py` and change:
```python
BLIP_USE_FINETUNED = True    # Use fine-tuned Telugu BLIP
YOLO_USE_CUSTOM    = True    # Use campus-trained YOLO
```

### Step 9 — Run the Application

```bash
python main.py
```

---

## 4. BLIP Telugu Training — In-Depth

```
Training data mix:
  IndicCaption (Telugu)  ████████████  60%   ← Primary language target
  VizWiz (English)       █████         25%   ← Blind user context
  COCO (English)         ███           15%   ← Grammar quality
```

After training, BLIP outputs Telugu sentences **natively** — e.g.:
> *"మీరు ఒక తరగతి గదిలో ఉన్నారు. మీకు ముందు ఒక కుర్చీ చాలా దగ్గరగా ఉంది."*
> ("You are in a classroom. A chair is very close ahead of you.")

---

## 5. YOLO Campus Training — What It Detects

After combined training:

| Category | Objects |
|----------|---------|
| People | person |
| Furniture | chair, couch, dining table, bed |
| Doors & Windows | door, openedDoor, cabinetDoor, window |
| Stairs & Ramps | stairs, ramp |
| Outdoor | bicycle, car, bus, motorcycle, bench, traffic light |
| Campus hazards | pole, corridor |
| General | 80 COCO classes (bag, bottle, cat, dog, etc.) |

---

## 6. Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | Batch too large | Use `--batch-size 2` |
| `FileNotFoundError: train_te.json` | IndicCaption not downloaded | `python data/download_datasets.py --dataset indic` |
| `FileNotFoundError: indoor_campus` | Manual datasets missing | See `data/MANUAL_DOWNLOADS.md` |
| No Telugu voice | edge-tts not installed | `pip install edge-tts pygame` |
| Translation fails | No internet | Check internet connection (edge-tts + deep-translator need it) |
| BLEU < 0.20 | Too few epochs | Add `--epochs 5` more |

---

## 7. Quick Reference

```bash
# Download all auto-downloadable data
python data/download_datasets.py --dataset all

# Train YOLO (campus navigation)
python training/train_detector.py --dataset combined --model yolo11s.pt

# Train BLIP (Telugu descriptions)
python training/train_captioner.py --epochs 8

# Evaluate
python training/evaluate.py

# Edit config.py: YOLO_USE_CUSTOM=True, BLIP_USE_FINETUNED=True

# Run
python main.py
```
