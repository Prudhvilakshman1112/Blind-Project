# Next Steps — mBLIP Campus Navigation System

> **Updated: June 2026** — Reflects full project state: mBLIP migration complete, LoRA training pipeline ready, FastAPI mobile server operational.

---

## Current Project State

| Component | Status | Action Required |
|-----------|--------|-----------------|
| YOLO Detection (18 campus classes) | ✅ Code ready | Download Roboflow datasets + train |
| mBLIP Telugu Captioner | ✅ Code ready | Create campus dataset + LoRA fine-tune |
| OCR (EasyOCR, Te+En) | ✅ Ready | No action needed |
| TTS (edge-tts Telugu voice) | ✅ Ready | Needs internet connection |
| FastAPI Server (`api.py`) | ✅ Ready | Run after training |
| Mobile Frontend (`frontend/index.html`) | ✅ Ready | Point to API server IP |
| Desktop App (`main.py`) | ✅ Ready | Run after training |
| Demo Mode (`demo/`) | ✅ Standalone | Needs Gemini API key only |

---

## Immediate Actions Required

### ⬅ Step 1 — Create Your Campus Caption Dataset (DO THIS FIRST)

Follow **[DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)** to:
- Take 300–500 photos on your college campus
- Write one Telugu caption per photo
- Format as `data/campus_captions/train.json` and `val.json`

```bash
# First, set up the folder structure:
python data/download_datasets.py --dataset campus-setup
```

> ⚠ **No public Telugu campus dataset exists.** `Hardik15/telugu-image-captions`, `FutureBeeAI/telugu-image-captions`, and `ai4bharat/IndicCOCO` are all deleted/private as of March 2026. You must collect your own campus photos.

---

### Step 2 — Download YOLO Campus Detection Datasets (Manual)

These 4 Roboflow/Kaggle datasets require a free account to download:

```bash
# See full download instructions:
python data/download_datasets.py --dataset manual-info
```

| Dataset | Source | Classes | Target Folder |
|---------|--------|---------|---------------|
| SCIN Indoor Navigation | [Roboflow](https://universe.roboflow.com/scin/indoor-navigation-system) | door, stairs | `data/indoor_campus/scin_indoor/` |
| Akhash Indoor Navigation | [Roboflow](https://universe.roboflow.com/akhash/indoor-navigation) | door, person, elevator, stair sign | `data/indoor_campus/akhash_indoor/` |
| Blind Indoor Navigation | [Roboflow](https://universe.roboflow.com) search: `IndoorNavigationForTheBlinds` | door, stairs, pole, chair, table | `data/indoor_campus/blind_indoor/` |
| Stairs Detection | [Kaggle](https://www.kaggle.com) search: `Stairs Detection YOLO Samuel Ayman` | stairs | `data/indoor_campus/stairs_kaggle/` |

> Export all Roboflow datasets as **YOLOv11 format**.

---

### Step 3 — Install All Dependencies

```bash
pip install -r requirements.txt
```

Key packages and what they do:

| Package | Purpose |
|---------|---------|
| `peft>=0.10.0` | LoRA adapter for mBLIP fine-tuning |
| `bitsandbytes>=0.43.0` | 4-bit NF4 quantization (RTX 3050 4 GB VRAM) |
| `transformers>=4.40.0` | Blip2Processor, Blip2ForConditionalGeneration |
| `ultralytics>=8.3.0` | YOLO11 training and inference |
| `edge-tts>=6.1.9` | Microsoft Neural Telugu voice (te-IN-ShrutiNeural) |
| `easyocr>=1.7.1` | Bilingual OCR (Telugu + English) |
| `deep-translator>=1.11.4` | English → Telugu translation for alert strings |
| `pygame>=2.5.0` | Audio playback for edge-tts |

> **Windows note:** `bitsandbytes` may not install on native Windows. If it fails, the code automatically falls back to float16 (needs 8 GB VRAM). Use WSL2 or Google Colab for 4-bit training.

---

### Step 4 — Verify Everything is Ready

```bash
python data/download_datasets.py --dataset verify
```

Expected output when ready to train:
```
✓ Campus captions : 400 train + 100 val pairs
✓ Campus detection: 4 sub-dataset(s) found
```

---

## Full Training Sequence (After Dataset is Ready)

```bash
# 1. Train YOLO11s campus object detector (18 classes)
python training/train_detector.py --dataset campus

# 2. Fine-tune mBLIP with LoRA on campus Telugu captions (3 epochs, ~100 MB adapter)
python training/train_captioner.py

# 3. Evaluate mBLIP (BLEU + METEOR metrics)
python training/evaluate.py

# 4. Activate trained models in config.py:
#    YOLO_USE_CUSTOM = True
#    MBLIP_USE_FINETUNED = True

# 5. Run the desktop application
python main.py

# 6. OR run the FastAPI server for mobile deployment
uvicorn api:app --host 0.0.0.0 --port 8000
```

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for the full 6-phase guide with VRAM settings and troubleshooting.

---

## Testing mBLIP Zero-Shot (Before Training)

You can test mBLIP's Telugu ability right now, before any fine-tuning:

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

proc  = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Gregor/mblip-mt0-xl", torch_dtype=torch.float16
).to("cuda")

img   = Image.open("your_campus_photo.jpg")
inp   = proc(images=img, text="క్లుప్తంగా వివరించు:", return_tensors="pt").to("cuda")
ids   = model.generate(**inp, max_new_tokens=25)
print(proc.decode(ids[0], skip_special_tokens=True))
```

> Note: The exact prompt used is `"క్లుప్తంగా వివరించు:"` (not English). See `config.py → MBLIP_PROMPT`.

---

## Running the Demo (No Training Required)

The `demo/` folder has a standalone demo using Gemini API:

```bash
# 1. Get a free Gemini API key from https://aistudio.google.com/apikey
# 2. Set it in demo/config_demo.py: GEMINI_API_KEY = "your_key_here"
# 3. Install demo dependencies:
pip install -r demo/requirements_demo.txt
# 4. Run:
python demo/main_demo.py
```

The demo uses `yolo11n.pt` (pre-trained, no campus fine-tuning) and Gemini for Telugu captions. Everything else (spatial reasoning, TTS, OCR, danger zones) is identical to the production app.

---

## Running the FastAPI Mobile Server

For mobile deployment (user's phone as camera):

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Open the mobile frontend
# On a phone on the same network, open:
# http://<your-PC-IP>:8000
# Or serve frontend/index.html via HTTP:
cd frontend && python -m http.server 8080
```

The frontend (`frontend/index.html`) captures frames at 8 FPS, POSTs to `/analyze`, and uses `window.speechSynthesis` (Web Speech API) with `te-IN` language for Telugu TTS on the mobile device.

---

## Cloud Training (Better Quality, Free)

If you want full float16 training (better quality, needs 12+ GB VRAM):

| Platform | VRAM | Cost | Command |
|---|---|---|---|
| Google Colab T4 | 15 GB | Free | `python training/train_captioner.py --no-4bit --batch-size 4` |
| Kaggle P100 | 16 GB | Free | Same |
| RunPod RTX 3090 | 24 GB | ~$0.4/hr | Same |

---

## After Training — Activate Models

Edit `config.py` to use your trained models:

```python
YOLO_USE_CUSTOM     = True   # Use checkpoints/yolo11_campus.pt
MBLIP_USE_FINETUNED = True   # Use checkpoints/mblip_campus/best/ LoRA adapter
```

Then run:
```bash
python main.py
```

The system will speak: *"Campus navigation system ready."* in Telugu and begin detecting objects.

---

## Key Configuration Flags (config.py)

| Flag | Default | Change When |
|------|---------|-------------|
| `MBLIP_USE_4BIT` | `True` | Set `False` for 8+ GB VRAM (better quality) |
| `MBLIP_USE_FINETUNED` | `False` | Set `True` after `train_captioner.py` completes |
| `YOLO_USE_CUSTOM` | `False` | Set `True` after `train_detector.py` completes |
| `TTS_ENGINE` | `"edge-tts"` | Set `"pyttsx3"` for fully offline English fallback |
| `TELUGU_MODE` | `True` | Set `False` for English-only testing |
| `SHOW_DEV_WINDOW` | `True` | Set `False` for headless/production mode |
