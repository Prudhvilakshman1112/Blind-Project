# Blind-Project — Campus Navigation Assistant for the Visually Impaired

**Real-time, Telugu-language assistive navigation for blind users on college/university campuses.**

> ⚠ **Documentation updated March 2025.** The project has been fully overhauled to use focused, campus-specific datasets. Old references to MS-COCO, VizWiz, and AI4Bharat IndicCOCO have been removed.

---

## What This Project Does

A Python application that:
1. **Detects** campus-relevant objects in real time (YOLO11 — 18 classes: doors, stairs, poles, people, chairs, etc.)
2. **Describes** the scene in **Telugu** (BLIP fine-tuned on Telugu image captions)
3. **Speaks** alerts and descriptions in Telugu via Microsoft Neural TTS (`te-IN-ShrutiNeural`)

**Priority system:**
- 🔴 **Stairs / Poles / Open Doors** → immediate audio interrupt ("Alert! Stairs very close, directly ahead!")
- 🟠 **Any object in danger zone** → high-priority alert
- 🟢 **General scene description** → spoken every 4 seconds

---

## Dataset Strategy (2025)

| Use | Dataset | Source | Size |
|-----|---------|--------|------|
| Telugu captions (BLIP) | `Hardik15/telugu-image-captions` | HuggingFace | ~25K pairs |
| Campus detection (YOLO) | SCIN Indoor Navigation | Roboflow | 544 images |
| Campus detection (YOLO) | Akhash Indoor Navigation | Roboflow | 1,115 images |
| Campus detection (YOLO) | Blind Indoor Navigation | Roboflow | manual |
| Campus detection (YOLO) | Stairs Detection | Kaggle | 1,000 images |

**Why NOT full COCO / VizWiz / AI4Bharat IndicCOCO:**
- MS-COCO: 18 GB, 80 classes — elephants, frisbees, pizzas — useless for campus and hurts accuracy
- VizWiz: Intentionally blurry/noisy blind-user photos — adds training noise without improvement
- AI4Bharat IndicCOCO: No longer available on HuggingFace (confirmed March 2025)

---

## Project Structure

```
Blind-Project/
├── main.py                    ← Main application entry point
├── config.py                  ← All settings (edit before training)
├── requirements.txt
│
├── src/
│   ├── vision_module.py       ← YOLO detection + spatial analysis
│   ├── caption_module.py      ← BLIP captioning + Telugu descriptions
│   ├── audio_module.py        ← Priority TTS (edge-tts / pyttsx3)
│   └── ocr_module.py          ← EasyOCR (auto-triggered + manual)
│
├── data/
│   ├── download_datasets.py   ← Download Telugu captions from HuggingFace
│   ├── dataset_loader.py      ← PyTorch Dataset classes
│   ├── augmentations.py       ← Image augmentation pipeline
│   ├── MANUAL_DOWNLOADS.md    ← Roboflow/Kaggle download instructions
│   ├── telugu_captions/       ← Auto-downloaded → train.json, val.json, images/
│   └── indoor_campus/         ← Manual download → Roboflow sub-folders
│
├── training/
│   ├── train_detector.py      ← Fine-tune YOLO11 on campus datasets
│   ├── train_captioner.py     ← Fine-tune BLIP on Telugu captions
│   ├── evaluate.py            ← Evaluate BLIP with BLEU/METEOR
│   └── export_models.py       ← Export to ONNX / OpenVINO
│
├── checkpoints/               ← Saved model weights
│   ├── yolo11_campus.pt       ← After YOLO training
│   └── blip_telugu/best/      ← After BLIP training
│
├── demo/                      ← Demo version (do NOT modify — separate system)
│   └── README_DEMO.md
│
└── TRAINING_GUIDE.md          ← Step-by-step training instructions
```

---

## Quick Start (Run Before Training)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Telugu Captions (Auto)
```bash
python data/download_datasets.py --dataset telugu
```

### 3. Download Campus Detection Datasets (Manual)
```bash
python data/download_datasets.py --dataset manual-info
```
Follow the instructions. Download 3–4 Roboflow datasets → place in `data/indoor_campus/`.

### 4. Verify
```bash
python data/download_datasets.py --dataset verify
```

---

## Training

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for full step-by-step instructions.

```bash
# Step 1: Train campus YOLO detector (needs manual datasets first)
python training/train_detector.py --dataset campus

# Step 2: Fine-tune BLIP on Telugu captions
python training/train_captioner.py

# Step 3: Evaluate
python training/evaluate.py
```

---

## Running the Application

Before running, set these flags in `config.py` after training:
```python
YOLO_USE_CUSTOM  = True    # Use trained campus YOLO weights
BLIP_USE_FINETUNED = True  # Use Telugu fine-tuned BLIP
```

```bash
# Live webcam
python main.py

# Test on a video file
python main.py --source path/to/video.mp4

# No audio (console output only)
python main.py --no-audio

# Headless (no developer window)
python main.py --no-window
```

### Keyboard Controls (Developer Window)
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Toggle manual OCR Reading Mode |
| `P` | Pause/Resume TTS |

---

## YOLO Classes (18 Campus-Specific)

| Category | Classes |
|----------|---------|
| People | `person` |
| Mobility | `bicycle`, `motorcycle`, `car` |
| Furniture | `bench`, `chair`, `table`, `backpack`, `laptop`, `cell phone` |
| Navigation | `door`, `openedDoor`, `window`, `stairs`, `step`, `ramp`, `pole`, `corridor` |

**High-Priority (immediate alert):** `stairs`, `step`, `ramp`, `openedDoor`, `pole`

---

## Configuration (config.py)

| Key Setting | Default | Notes |
|-------------|---------|-------|
| `YOLO_MODEL_NAME` | `yolo11s.pt` | Use `yolo11m.pt` for +8% mAP if 8GB+ VRAM |
| `YOLO_USE_CUSTOM` | `False` | Set `True` after campus training |
| `BLIP_USE_FINETUNED` | `False` | Set `True` after Telugu training |
| `TELUGU_MODE` | `True` | All output in Telugu |
| `TTS_ENGINE` | `edge-tts` | Set `pyttsx3` for fully offline use |
| `DANGER_ZONE_X_RATIO` | `0.28` | 28% width centre band |
| `DANGER_ZONE_Y_RATIO` | `0.40` | 40% height centre band |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 6–8 GB |
| RAM | 8 GB | 16 GB |
| Internet | Required for edge-tts | — |
| Camera | USB webcam | 720p+ |

---

## Demo Version

The `demo/` folder contains a separate demo application for testing without training.
**Do not modify demo files** — they are a standalone system independent of this main project.
See `demo/README_DEMO.md` for demo instructions.
