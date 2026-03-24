# Training Guide — Blind-Project Campus Navigation

> **Updated March 2026** — Captioner migrated from BLIP to **mBLIP** (multilingual BLIP-2).
> mBLIP already speaks Telugu — LoRA fine-tuning on your campus photos only takes 3 epochs.

This guide walks you through the full training pipeline from dataset preparation to a deployable model.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure you have:
- Python 3.10+
- PyTorch with CUDA (check: `python -c "import torch; print(torch.cuda.is_available())"`)
- 4 GB GPU VRAM minimum (RTX 3050 works with 4-bit quantization)
- Internet connection (for mBLIP download ~5 GB on first run + edge-tts TTS)

---

## Phase 1 — Prepare Datasets

### Step 1A: Campus Caption Dataset (for mBLIP)

**This is your own photos + Telugu captions.** mBLIP already knows Telugu — you just teach it your campus.

```bash
# Create the folder structure
python data/download_datasets.py --dataset campus-setup
```

This creates `data/campus_captions/` with sample JSON files.

**Then follow [DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)** for:
- What photos to take (300–500 campus photos)
- How to write Telugu captions
- The exact JSON format required
- How to split into train/val sets

Minimum dataset structure when ready:
```
data/campus_captions/
├── train.json    ← 240–400 Telugu caption pairs
├── val.json      ← 60–100 Telugu caption pairs
└── images/       ← your .jpg campus photos
```

### Step 1B: Campus Detection Datasets (Manual — for YOLO)

Show full instructions:
```bash
python data/download_datasets.py --dataset manual-info
```

Download these **4 datasets** from Roboflow/Kaggle:

| Dataset | Source | Target Folder |
|---------|--------|---------------|
| SCIN Indoor Navigation | [Roboflow](https://universe.roboflow.com/scin/indoor-navigation-system) | `data/indoor_campus/scin_indoor/` |
| Akhash Indoor Navigation | [Roboflow](https://universe.roboflow.com/akhash/indoor-navigation) | `data/indoor_campus/akhash_indoor/` |
| Blind Indoor Navigation | Roboflow (search: IndoorNavigationForTheBlinds) | `data/indoor_campus/blind_indoor/` |
| Stairs Detection | Kaggle (search: Stairs Detection YOLO Samuel Ayman) | `data/indoor_campus/stairs_kaggle/` |

**Always export Roboflow datasets as YOLOv11 format.**

### Step 1C: Verify Everything
```bash
python data/download_datasets.py --dataset verify
```

Expected output:
```
✓ Campus captions : 400 train + 100 val pairs
✓ Campus detection: 4 sub-dataset(s) found
```

---

## Phase 2 — Train YOLO Campus Detector

Fine-tunes YOLO11s on 18 campus-only classes (no irrelevant COCO objects).

```bash
# Standard (4-6 GB VRAM):
python training/train_detector.py --dataset campus

# Higher accuracy (8+ GB VRAM):
python training/train_detector.py --dataset campus --model yolo11m.pt

# Tune epochs if needed:
python training/train_detector.py --dataset campus --epochs 100
```

**Training output:**
- Progress logged to `logs/yolo_training.log` and `logs/yolo_runs/`
- Best weights automatically saved to `checkpoints/yolo11_campus.pt`

**After training completes:**
Edit `config.py`:
```python
YOLO_USE_CUSTOM = True
```

Expected results (yolo11s, 80 epochs): **mAP50 ≥ 0.60** on combined campus test set.

---

## Phase 3 — Fine-Tune mBLIP on Telugu Campus Captions

This teaches mBLIP campus-specific Telugu vocabulary using **LoRA** (Low-Rank Adaptation).

> **mBLIP already speaks Telugu.** LoRA only adds campus-specific knowledge.
> Only the LoRA adapter is saved (~100 MB) — not the full 5 GB model.

```bash
# RTX 3050 4 GB (4-bit quantization, default):
python training/train_captioner.py

# Fewer epochs (faster test run):
python training/train_captioner.py --epochs 1

# For cloud/Colab with 12+ GB VRAM (no 4-bit, better quality):
python training/train_captioner.py --no-4bit --batch-size 4

# Resume from checkpoint:
python training/train_captioner.py --resume checkpoints/mblip_campus/checkpoint_epoch2
```

**Training output:**
- Progress logged to `logs/mblip_training.log`
- LoRA adapter checkpoints: `checkpoints/mblip_campus/checkpoint_epochN/`
- Best adapter: `checkpoints/mblip_campus/best/`

**After training completes:**
Edit `config.py`:
```python
MBLIP_USE_FINETUNED = True
```

### LoRA vs Full Model Training
| | LoRA (default) | Full fine-tuning |
|---|---|---|
| Parameters trained | ~1% (~50 MB) | 100% (~5 GB) |
| VRAM needed | 4 GB (with 4-bit) | 16 GB+ |
| Epochs needed | 3 | 8–15 |
| Saved model size | ~100 MB | ~5 GB |
| Telugu quality | Excellent | Excellent |

---

## Phase 4 — Evaluate

```bash
# Evaluate on campus val split (BLEU + METEOR)
python training/evaluate.py

# Evaluate zero-shot (no LoRA adapter — base mBLIP only)
python training/evaluate.py --model Gregor/mblip-mt0-xl

# Limit samples for speed
python training/evaluate.py --max-samples 50
```

**Target metrics:**
- BLEU-4 ≥ 0.25
- METEOR ≥ 0.28

Results saved to `logs/eval_report.json`.

---

## Phase 5 — Run the Application

```bash
python main.py
```

Or with a video file for testing:
```bash
python main.py --source path/to/video.mp4
```

---

## Phase 6 — Export (Optional, for Edge Deployment)

```bash
python training/export_models.py
```

Exports YOLO to ONNX / OpenVINO for faster inference on CPU.
Note: mBLIP export to ONNX requires additional steps — see `training/export_models.py`.

---

## Tips for Low-VRAM GPUs (4 GB RTX 3050)

These are already configured as defaults in `config.py`:
```python
MBLIP_USE_4BIT         = True   # 4-bit quantization
MBLIP_TRAIN_BATCH_SIZE = 1      # Batch size 1
MBLIP_GRAD_ACCUM_STEPS = 8      # Effective batch = 8
MBLIP_NUM_BEAMS        = 3      # Reduced beam search
```

---

## Cloud Training (Recommended for Better Quality)

To train without quantization on free cloud GPUs:

**Google Colab (T4, 15 GB free):**
1. Upload your campus_captions dataset
2. Clone this repo in Colab
3. Run: `python training/train_captioner.py --no-4bit --batch-size 4`

**Kaggle Notebooks (P100, 16 GB free):**
Same approach. Enable GPU in notebook settings.

---

## Common Issues

| Error | Fix |
|-------|-----|
| CUDA out of memory | Ensure `MBLIP_USE_4BIT = True` in config.py; reduce `MBLIP_TRAIN_BATCH_SIZE` to 1 |
| `Campus captions not found` | Run `python data/download_datasets.py --dataset campus-setup` then add your photos |
| `No campus datasets found` | Follow `data/MANUAL_DOWNLOADS.md` for YOLO datasets |
| `bitsandbytes not installed` | Try: `pip install bitsandbytes`. On Windows without WSL2, float16 fallback is used |
| `peft not found` | Run: `pip install peft>=0.10.0` |
| edge-tts timeout | Check internet connection; set `TTS_ENGINE = "pyttsx3"` for offline fallback |
| YOLO model not found | Run `python training/train_detector.py` first or set `YOLO_USE_CUSTOM = False` |
| mBLIP 404 error | Check internet; model downloads from HuggingFace (~5 GB, one time) |
