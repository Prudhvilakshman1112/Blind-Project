# Blind-Project — Campus Navigation Assistant for the Visually Impaired

**Real-time, Telugu-language assistive navigation for blind users on college/university campuses.**

> ⚠ **Documentation updated March 2026.** The captioning module has been migrated from standard BLIP fine-tuning to **mBLIP** (multilingual BLIP-2), which natively supports Telugu. See the [mBLIP section](#mblip-architecture) below.

---

## What This Project Does

A Python application that:
1. **Detects** campus-relevant objects in real time (YOLO11 — 18 classes: doors, stairs, poles, people, chairs, etc.)
2. **Describes** the scene in **Telugu natively** (mBLIP — multilingual BLIP-2 supporting 96 languages including Telugu)
3. **Speaks** alerts and descriptions in Telugu via Microsoft Neural TTS (`te-IN-ShrutiNeural`)

**Priority system:**
- 🔴 **Stairs / Poles / Open Doors** → immediate audio interrupt ("హెచ్చరిక! మెట్లు చాలా దగ్గరగా!")
- 🟠 **Any object in danger zone** → high-priority alert ("జాగ్రత్త! ...")
- 🟢 **General scene description** → spoken every 4 seconds (in Telugu, from mBLIP)

---

## mBLIP Architecture

The captioning module uses **mBLIP** (`Gregor/mblip-mt0-xl`):
- Multilingual BLIP-2 model trained on **96 languages including Telugu**
- Describes campus scenes **in Telugu natively** — no translation step needed
- ~5 GB download on first run (cached, only once)
- Optional LoRA fine-tuning on your own campus photos for domain-specific vocabulary

| | Old (standard BLIP) | New (mBLIP) |
|---|---|---|
| Telugu ability | Needed fine-tuning on deleted dataset | ✅ Native Telugu from day 1 |
| Dataset needed | 25,000+ Telugu pairs (unavailable) | 300–500 campus photos (you create) |
| VRAM (4 GB RTX 3050) | ✓ Fine-tuning OK | ✓ 4-bit quantization enabled |
| Caption quality | Depended on dataset quality | Multilingual model, high quality |

---

## Dataset Strategy (March 2026)

| Use | Dataset | Source | Size |
|-----|---------|--------|------|
| Scene captions (mBLIP) | **Your own campus photos + Telugu captions** | You collect | 300–500 images |
| Campus detection (YOLO) | SCIN Indoor Navigation | Roboflow | 544 images |
| Campus detection (YOLO) | Akhash Indoor Navigation | Roboflow | 1,115 images |
| Campus detection (YOLO) | Blind Indoor Navigation | Roboflow | manual |
| Campus detection (YOLO) | Stairs Detection | Kaggle | 1,000 images |

**See [DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md) for full instructions on creating your campus caption dataset.**

---

## Project Structure

```
Blind-Project/
├── main.py                    ← Main application entry point
├── config.py                  ← All settings (edit before training)
├── requirements.txt
├── DATASET_CREATION_GUIDE.md  ← HOW TO CREATE YOUR CAMPUS DATASET ← START HERE
│
├── src/
│   ├── vision_module.py       ← YOLO detection + spatial analysis
│   ├── caption_module.py      ← mBLIP captioning (Telugu native)
│   ├── audio_module.py        ← Priority TTS (edge-tts / pyttsx3)
│   └── ocr_module.py          ← EasyOCR (auto-triggered + manual)
│
├── data/
│   ├── download_datasets.py   ← Setup campus caption folder structure
│   ├── dataset_loader.py      ← PyTorch Dataset classes (mBLIP)
│   ├── augmentations.py       ← Image augmentation pipeline
│   ├── MANUAL_DOWNLOADS.md    ← Roboflow/Kaggle download instructions
│   ├── campus_captions/       ← YOUR campus photos + Telugu captions
│   │   ├── train.json             ← Training caption pairs
│   │   ├── val.json               ← Validation caption pairs
│   │   └── images/                ← Your campus .jpg photos
│   └── indoor_campus/         ← Manual download → Roboflow sub-folders
│
├── training/
│   ├── train_detector.py      ← Fine-tune YOLO11 on campus datasets
│   ├── train_captioner.py     ← Fine-tune mBLIP with LoRA
│   ├── evaluate.py            ← Evaluate mBLIP with BLEU/METEOR
│   └── export_models.py       ← Export to ONNX / OpenVINO
│
├── checkpoints/               ← Saved model weights
│   ├── yolo11_campus.pt       ← After YOLO training
│   └── mblip_campus/best/     ← LoRA adapter after mBLIP training
│
├── demo/                      ← Demo version (do NOT modify)
└── TRAINING_GUIDE.md          ← Step-by-step training instructions
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Campus Caption Dataset Folder
```bash
python data/download_datasets.py --dataset campus-setup
```
Then follow **[DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)** to add your campus photos.

### 3. Download Campus Detection Datasets (Manual — for YOLO)
```bash
python data/download_datasets.py --dataset manual-info
```
Follow instructions. Download 3–4 Roboflow datasets → place in `data/indoor_campus/`.

### 4. Verify Everything is Ready
```bash
python data/download_datasets.py --dataset verify
```

---

## Training

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for full step-by-step instructions.

```bash
# Step 1: Train campus YOLO detector (needs Roboflow datasets first)
python training/train_detector.py --dataset campus

# Step 2: Fine-tune mBLIP on your campus captions (LoRA — only ~100 MB adapter)
python training/train_captioner.py

# Step 3: Evaluate mBLIP on your campus val set
python training/evaluate.py
```

---

## Running the Application

Before running, set these flags in `config.py` after training:
```python
YOLO_USE_CUSTOM    = True    # Use trained campus YOLO weights
MBLIP_USE_FINETUNED = True  # Use campus LoRA adapter
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

**High-Priority (immediate interrupt alert):** `stairs`, `step`, `ramp`, `openedDoor`, `pole`

---

## Configuration (config.py)

| Key Setting | Default | Notes |
|-------------|---------|-------|
| `MBLIP_PRETRAINED_NAME` | `Gregor/mblip-mt0-xl` | Base mBLIP model |
| `MBLIP_USE_FINETUNED` | `False` | Set `True` after campus LoRA training |
| `MBLIP_USE_4BIT` | `True` | 4-bit quantization for 4 GB VRAM (RTX 3050) |
| `MBLIP_PROMPT` | `"Describe this campus scene in Telugu:"` | Telugu output trigger |
| `YOLO_MODEL_NAME` | `yolo11s.pt` | Use `yolo11m.pt` for +8% mAP if 8GB+ VRAM |
| `YOLO_USE_CUSTOM` | `False` | Set `True` after campus training |
| `TELUGU_MODE` | `True` | All output in Telugu |
| `TTS_ENGINE` | `edge-tts` | Set `pyttsx3` for fully offline use |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB (with MBLIP_USE_4BIT=True) | 8 GB+ (for full float16) |
| RAM | 8 GB | 16 GB |
| Internet | Required (mBLIP download + edge-tts) | — |
| Camera | USB webcam | 720p+ |

> **For cloud training:** Google Colab (T4, 15 GB VRAM) or Kaggle (P100, 16 GB VRAM) are free options. Set `MBLIP_USE_4BIT = False` for full-quality training on cloud.

---

## Demo Version

The `demo/` folder contains a separate demo application for testing without training.
**Do not modify demo files** — they are a standalone system independent of this main project.
See `demo/README_DEMO.md` for demo instructions.
