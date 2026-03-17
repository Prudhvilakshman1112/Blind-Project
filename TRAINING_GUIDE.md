# Training Guide — Blind-Project Campus Navigation

> **Updated March 2025** — Complete overhaul. Old COCO / VizWiz / IndicCOCO steps removed.

This guide walks you through the full training pipeline from dataset download to a deployable model.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure you have:
- Python 3.10+
- PyTorch with CUDA (check: `python -c "import torch; print(torch.cuda.is_available())"`)
- At least 4 GB GPU VRAM (6–8 GB recommended)
- Internet connection (for HuggingFace download + edge-tts TTS)

---

## Phase 1 — Download Datasets

### Step 1A: Telugu Captions (Auto)
Downloads `Hardik15/telugu-image-captions` from HuggingFace (~25K image-caption pairs).

```bash
python data/download_datasets.py --dataset telugu
```

This saves to:
```
data/telugu_captions/
├── train.json    ← ~22,500 Telugu caption pairs
├── val.json      ← ~2,500 Telugu caption pairs
└── images/       ← downloaded image files
```

### Step 1B: Campus Detection Datasets (Manual)

Show full instructions:
```bash
python data/download_datasets.py --dataset manual-info
```

Download these **4 datasets** from Roboflow/Kaggle and put them in the correct folders:

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

---

## Phase 2 — Train YOLO Campus Detector

This fine-tunes YOLO11s on 18 campus-only classes (no irrelevant COCO objects).

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
- Training plots saved in `logs/yolo_runs/yolo11_campus_yolo11s/`

**After training completes:**
Edit `config.py`:
```python
YOLO_USE_CUSTOM = True
```

Expected results (yolo11s, 80 epochs): **mAP50 ≥ 0.60** on combined campus test set.

---

## Phase 3 — Fine-Tune BLIP on Telugu Captions

This teaches BLIP to generate scene descriptions in **native Telugu**.

```bash
# Standard (4 GB VRAM):
python training/train_captioner.py

# Faster iteration (less VRAM):
python training/train_captioner.py --batch-size 2 --epochs 5

# Resume from checkpoint:
python training/train_captioner.py --resume checkpoints/blip_telugu/checkpoint_epoch3
```

**Training output:**
- Progress logged to `logs/blip_training.log`
- Epoch checkpoints: `checkpoints/blip_telugu/checkpoint_epochN/`
- Best model: `checkpoints/blip_telugu/best/`

**After training completes:**
Edit `config.py`:
```python
BLIP_USE_FINETUNED = True
```

**Note:** Once BLIP is fine-tuned on Telugu, the audio module can pipe BLIP output directly to edge-tts — no translation step needed. The `deep-translator` step is still used for English danger alerts.

---

## Phase 4 — Evaluate

```bash
# Evaluate on Telugu val split (BLEU + METEOR)
python training/evaluate.py

# Limit evaluation samples for speed
python training/evaluate.py --max-samples 500
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

Exports BLIP vision encoder to ONNX / OpenVINO for faster inference on CPU.

---

## Tips for Low-VRAM GPUs (4 GB)

In `config.py`:
```python
BLIP_TRAIN_BATCH_SIZE = 2     # Reduce from 4
BLIP_GRAD_ACCUM_STEPS = 8     # Increase to compensate (effective batch = 16)
YOLO_TRAIN_BATCH_SIZE = 8     # Reduce from 16
```

---

## Common Issues

| Error | Fix |
|-------|-----|
| CUDA out of memory | Reduce `BLIP_TRAIN_BATCH_SIZE` to 2 in config.py |
| `Telugu captions not found` | Run `python data/download_datasets.py --dataset telugu` |
| `No campus datasets found` | Follow `data/MANUAL_DOWNLOADS.md` |
| edge-tts timeout | Check internet connection; set `TTS_ENGINE = "pyttsx3"` for offline |
| YOLO model not found | Run `python training/train_detector.py` first or set `YOLO_USE_CUSTOM = False` |
