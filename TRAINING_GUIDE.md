# Training Guide — Blind-Project Campus Navigation

> **Updated: June 2026** — Captioner uses **mBLIP** (`Gregor/mblip-mt0-xl`, multilingual BLIP-2).
> mBLIP already speaks Telugu natively — LoRA fine-tuning on campus photos takes only 3 epochs.
> YOLO11s detector trained on 18 campus-specific classes (no irrelevant COCO objects).

This guide walks you through the full training pipeline from dataset preparation to a deployable model.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Ensure you have:
- **Python 3.10+**
- **PyTorch with CUDA** — check: `python -c "import torch; print(torch.cuda.is_available())"`
- **4 GB GPU VRAM minimum** — RTX 3050 works with 4-bit quantization (`MBLIP_USE_4BIT=True`)
- **Internet connection** — for mBLIP model download (~5 GB, one time) + edge-tts TTS calls

> **Windows note:** `bitsandbytes` (4-bit quantization) may not install cleanly on native Windows. If `pip install bitsandbytes` fails, the code automatically falls back to float16 (requires ~8 GB VRAM). Use WSL2 or Google Colab for 4-bit training on Windows hardware.

---

## Phase 1 — Prepare Datasets

### Step 1A: Campus Caption Dataset (for mBLIP fine-tuning)

**This is your own photos + Telugu captions.** mBLIP already knows Telugu — you just teach it your campus.

```bash
# Create the folder structure
python data/download_datasets.py --dataset campus-setup
```

This creates `data/campus_captions/` with sample JSON files (2 train, 1 val — placeholders only).

**Then follow [DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)** for:
- What photos to take (300–500 campus photos)
- How to write Telugu captions (one sentence per photo)
- The exact JSON format required by `CampusCaptionDataset`
- How to split into train/val sets (80/20)

Minimum dataset structure when ready:
```
data/campus_captions/
├── train.json    ← 240–400 Telugu caption pairs (JSON array)
├── val.json      ← 60–100 Telugu caption pairs  (JSON array)
└── images/       ← your .jpg campus photos
```

Each JSON entry format:
```json
{ "idx": 0, "file_name": "train_000000.jpg", "caption": "Telugu caption here" }
```

### Step 1B: Campus Detection Datasets (Manual — for YOLO)

Show full download instructions:
```bash
python data/download_datasets.py --dataset manual-info
```

Download these **4 datasets** from Roboflow/Kaggle (free account required):

| Dataset | Source | Classes | Target Folder |
|---------|--------|---------|---------------|
| SCIN Indoor Navigation | [Roboflow](https://universe.roboflow.com/scin/indoor-navigation-system) | door, stairs | `data/indoor_campus/scin_indoor/` |
| Akhash Indoor Navigation | [Roboflow](https://universe.roboflow.com/akhash/indoor-navigation) | door, person, elevator, stair sign | `data/indoor_campus/akhash_indoor/` |
| Blind Indoor Navigation | Roboflow (search: `IndoorNavigationForTheBlinds`) | door, stairs, pole, chair, table | `data/indoor_campus/blind_indoor/` |
| Stairs Detection | Kaggle (search: `Stairs Detection YOLO Samuel Ayman`) | stairs | `data/indoor_campus/stairs_kaggle/` |

**Always export Roboflow datasets as YOLOv11 format.**

The YOLO training script (`training/train_detector.py`) auto-discovers all sub-folders inside `data/indoor_campus/` and trains on the combined dataset. Supported Roboflow export structures:
- `images/train/` + `images/valid/` (standard)
- `train/images/` + `valid/images/` (alternative)

### Step 1C: Verify Everything

```bash
python data/download_datasets.py --dataset verify
```

Expected output when both datasets are ready:
```
✓ Campus captions : 400 train + 100 val pairs
✓ Campus detection: 4 sub-dataset(s) found
```

---

## Phase 2 — Train YOLO Campus Detector

Fine-tunes YOLO11s on 18 campus-only classes (no irrelevant COCO objects like elephants or pizzas).

**Campus classes (18 total):**
```
person, bicycle, motorcycle, car,
bench, chair, table, backpack, laptop, cell phone,
door, openedDoor, window, stairs, step, ramp, pole, corridor
```

**High-priority classes (immediate TTS interrupt):** `stairs`, `step`, `ramp`, `openedDoor`, `pole`

```bash
# Standard training (4–6 GB VRAM):
python training/train_detector.py --dataset campus

# Higher accuracy (8+ GB VRAM, ~8% higher mAP):
python training/train_detector.py --dataset campus --model yolo11m.pt

# Custom epochs:
python training/train_detector.py --dataset campus --epochs 100
```

**Training configuration** (from `config.py`):
```python
YOLO_TRAIN_EPOCHS     = 80
YOLO_TRAIN_BATCH_SIZE = 16
YOLO_TRAIN_IMG_SIZE   = 640
YOLO_TRAIN_LR0        = 0.01
YOLO_TRAIN_PATIENCE   = 15    # Early stopping
```

**Training output:**
- Progress logged to `logs/yolo_training.log` and `logs/yolo_runs/`
- Best weights automatically saved to `checkpoints/yolo11_campus.pt`
- Full YOLO run artifacts in `logs/yolo_runs/yolo11_campus_yolo11s/`

**After training completes:**
Edit `config.py`:
```python
YOLO_USE_CUSTOM = True
```

Expected results (yolo11s, 80 epochs on combined campus datasets): **mAP50 ≥ 0.60**

---

## Phase 3 — Fine-Tune mBLIP on Telugu Campus Captions

This teaches mBLIP campus-specific Telugu vocabulary using **LoRA** (Low-Rank Adaptation).

> **mBLIP already speaks Telugu.** LoRA only adds campus-specific knowledge (doors, corridors, specific campus layout descriptions).
> Only the LoRA adapter is saved (~100 MB) — not the full 5 GB mBLIP model.

### LoRA Architecture
- LoRA wraps only the **language model** (LM) layers of mBLIP
- Vision encoder and Q-Former stay **frozen**
- Target modules: `["q", "v"]` (query and value attention matrices)
- LoRA rank: `r=16`, alpha: `32`, dropout: `0.05` (see `config.py`)

### Running Training

```bash
# RTX 3050 4 GB (4-bit quantization, default):
python training/train_captioner.py

# Fewer epochs (faster test run — verifies pipeline works):
python training/train_captioner.py --epochs 1

# For cloud/Colab with 12+ GB VRAM (float16, better quality):
python training/train_captioner.py --no-4bit --batch-size 4

# Resume from checkpoint after interruption:
python training/train_captioner.py --resume checkpoints/mblip_campus/checkpoint_epoch2
```

### Training configuration (from `config.py`)

```python
MBLIP_TRAIN_EPOCHS     = 3       # Usually enough for LoRA
MBLIP_TRAIN_BATCH_SIZE = 1       # RTX 3050 4 GB: must be 1
MBLIP_LEARNING_RATE    = 2e-4    # LoRA standard learning rate
MBLIP_GRAD_ACCUM_STEPS = 8       # Effective batch = 1 × 8 = 8
MBLIP_WARMUP_STEPS     = 50
MBLIP_LORA_RANK        = 16
MBLIP_LORA_ALPHA       = 32
MBLIP_LORA_DROPOUT     = 0.05
```

### Training output
- Progress logged to `logs/mblip_training.log`
- LoRA adapter checkpoints: `checkpoints/mblip_campus/checkpoint_epochN/`
- Best adapter (lowest val loss): `checkpoints/mblip_campus/best/`

### After training completes

Edit `config.py`:
```python
MBLIP_USE_FINETUNED = True
```

### LoRA vs Full Model Comparison

| | LoRA (default) | Full fine-tuning |
|---|---|---|
| Parameters trained | ~1% (~50 MB adapter) | 100% (~5 GB) |
| VRAM needed | 4 GB (with 4-bit) | 16 GB+ |
| Epochs needed | 3 | 8–15 |
| Saved adapter size | ~100 MB | ~5 GB |
| Telugu quality | Excellent (campus-tuned) | Excellent |
| Training time | ~30 min on RTX 3050 | Hours |

---

## Phase 4 — Evaluate mBLIP

Evaluates the base or LoRA fine-tuned mBLIP using BLEU-1/2/3/4 and METEOR metrics.

```bash
# Evaluate LoRA adapter (default: checkpoints/mblip_campus/best/)
python training/evaluate.py

# Evaluate a specific checkpoint:
python training/evaluate.py --model checkpoints/mblip_campus/checkpoint_epoch2

# Zero-shot evaluation (base mBLIP, no LoRA adapter):
python training/evaluate.py --model Gregor/mblip-mt0-xl

# Limit samples for quick evaluation:
python training/evaluate.py --max-samples 50

# For cloud GPUs (no 4-bit):
python training/evaluate.py --no-4bit
```

**Target metrics:**
- BLEU-4 ≥ 0.25
- METEOR ≥ 0.28

Detailed results (all samples with predicted vs reference captions) saved to `logs/eval_report.json`.

> Note: BLEU/METEOR for Telugu may be lower than English benchmarks due to morphological complexity. Focus on whether the output is **navigationally useful**, not just numerically high.

---

## Phase 5 — Run the Application

### Option A: Desktop Application (main.py)

```bash
# Live webcam (default camera)
python main.py

# Specific camera index
python main.py --source 1

# Test on a video file (no camera needed)
python main.py --source path/to/video.mp4

# Headless (no OpenCV window — for server/background)
python main.py --no-window

# Disable TTS (console output only — for debugging)
python main.py --no-audio
```

**Keyboard controls in developer window:**
| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Toggle manual OCR reading mode |
| `P` | Pause / resume TTS descriptions |

**What the developer window shows:**
- Bounding boxes (Green=normal, Orange=high-priority, Red=danger zone)
- Clock position and distance word for each detection
- FPS counter
- `DANGER ZONE` rectangle (centre 28% × 40% of frame)
- `[READING MODE]` indicator when OCR is active
- Current mBLIP caption (bottom bar)

### Option B: FastAPI Mobile Server (api.py)

For mobile deployment where the user's phone is the camera:

```bash
# Start the backend API
uvicorn api:app --host 0.0.0.0 --port 8000

# Serve the frontend on the same machine
cd frontend && python -m http.server 8080
```

The frontend (`frontend/index.html`) connects to `http://localhost:8000/analyze`, captures frames at 8 FPS using the phone camera, and uses `window.speechSynthesis` with `te-IN` for Telugu TTS.

> To access from a phone on the same network, replace `localhost` in `frontend/index.html` with your PC's local IP address (e.g., `192.168.1.x`).

---

## Phase 6 — Export Models (Optional, for Edge Deployment)

Exports models to ONNX and OpenVINO IR format for CPU deployment on lower-power devices:

```bash
python training/export_models.py

# Skip YOLO export (only export BLIP):
python training/export_models.py --skip-yolo

# Skip BLIP export (only export YOLO):
python training/export_models.py --skip-blip
```

**What gets exported:**
1. BLIP Vision Encoder → ONNX (`exported_models/mblip_vision_encoder.onnx`)
2. BLIP Vision Encoder → OpenVINO IR (`exported_models/mblip_openvino/`)
3. YOLO11 → ONNX (via ultralytics built-in export, dynamic axes)
4. YOLO11 → OpenVINO IR (fp16, via ultralytics built-in export)

> ONNX and OpenVINO exports enable inference on devices without dedicated GPUs (Intel iGPU, Raspberry Pi, Jetson Nano).

---

## Tips for Low-VRAM GPUs (4 GB RTX 3050)

These are already configured as defaults in `config.py`:
```python
MBLIP_USE_4BIT         = True    # 4-bit NF4 quantization (bitsandbytes)
MBLIP_TRAIN_BATCH_SIZE = 1       # Batch size 1 (only viable option at 4 GB)
MBLIP_GRAD_ACCUM_STEPS = 8       # Effective batch = 8 without extra VRAM
MBLIP_NUM_BEAMS        = 1       # Greedy search (fastest inference)
MBLIP_MAX_NEW_TOKENS   = 25      # Short captions (navigation, not poetry)
```

During inference, an additional optimization in `caption_module.py`:
- Image is resized to **224×224** before encoding (reduces GPU memory)
- Scene change detection (`SCENE_CHANGE_THRESHOLD = 15.0`) skips inference if the user is standing still

---

## Cloud Training (Recommended for Better Quality)

To train without quantization on free cloud GPUs:

**Google Colab (T4, 15 GB VRAM, free):**
1. Upload your `data/campus_captions/` dataset
2. Clone this repo in Colab
3. Run: `python training/train_captioner.py --no-4bit --batch-size 4`

**Kaggle Notebooks (P100, 16 GB VRAM, free):**
Same approach. Enable GPU in notebook settings → Settings → Accelerator → GPU P100.

---

## Common Issues

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Ensure `MBLIP_USE_4BIT = True` in config.py; set `MBLIP_TRAIN_BATCH_SIZE = 1` |
| `Campus captions not found` | Run `python data/download_datasets.py --dataset campus-setup` then add your photos |
| `No campus datasets found` | Follow `data/MANUAL_DOWNLOADS.md` for YOLO Roboflow datasets |
| `bitsandbytes not installed` | Try: `pip install bitsandbytes`. On Windows, float16 fallback is auto-used (needs 8 GB VRAM) |
| `peft not found` | Run: `pip install peft>=0.10.0` |
| `edge-tts timeout / SSLError` | Check internet connection; set `TTS_ENGINE = "pyttsx3"` in config.py for offline fallback |
| `YOLO model not found` | Run `python training/train_detector.py` first OR set `YOLO_USE_CUSTOM = False` to use base YOLO |
| `mBLIP 404 error / OSError` | Check internet; model auto-downloads from HuggingFace (~5 GB, cached after first run) |
| `Only sample data in val.json` | Follow DATASET_CREATION_GUIDE.md — you must add your own campus photos and replace sample JSON |
| `easyocr model not found` | First run downloads EasyOCR models (~100 MB) automatically — needs internet |
| `deep_translator ImportError` | Run: `pip install deep-translator` (for English alert translation to Telugu) |
