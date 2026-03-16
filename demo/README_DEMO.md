# 🦯 Blind-Project — DEMO MODE

> **Purpose:** Preview the full application experience **without training any models**.
> Uses Google Gemini 1.5 Flash API for scene captioning + pre-trained YOLOv11n.
> Once you're happy with the output, switch to the main project for full training.

---

## What is Different from the Final App?

| Component | DEMO | Final App |
|-----------|------|-----------|
| Scene captioning | Gemini 1.5 Flash API (free) | Fine-tuned BLIP (local) |
| Object detection | YOLOv11n pre-trained | YOLOv11n fine-tuned |
| Everything else | **IDENTICAL** | **IDENTICAL** |

All spatial reasoning, TTS voice, OCR, developer window, and keyboard controls
behave **exactly the same** as the final application will.

---

## 3-Step Setup

### Step 1 — Get a Free Gemini API Key

1. Go to: **https://aistudio.google.com/apikey**
2. Sign in with your Google account (free — no credit card needed)
3. Click **"Create API Key"** → copy the key

Free tier limits: **15 requests/minute, 1,500 requests/day** ✅
(The demo calls Gemini once every 4 seconds = 15 calls/minute max — stays within limit)

---

### Step 2 — Set Your API Key

Open `demo/config_demo.py` and replace the placeholder:

```python
# Line 13 in config_demo.py — change this:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# to your actual key, e.g.:
GEMINI_API_KEY = "AIzaSyAbc123XYZ..."
```

---

### Step 3 — Install & Run

```bash
# Make sure your virtual environment is active
.venv\Scripts\activate

# Install demo dependencies (includes google-generativeai)
pip install -r demo/requirements_demo.txt

# Run the demo!
python demo/main_demo.py
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit demo |
| `R` | Toggle OCR / Reading Mode (reads text aloud) |
| `P` | Pause / resume voice descriptions |

---

## Developer Window Guide

```
┌───────────────────────────────────────────────────────────────┐
│ FPS: 28.4  [DEMO MODE — Gemini Flash]                         │
│                                                               │
│  ┌──────────────────────────┐                                 │
│  │ DANGER ZONE (red rect)   │ ← centre 35%×50% of frame      │
│  │  [person  92% | directly ahead of you | about 1 metre away]  ← RED box   │
│  │  [chair   78% | to your left          | about 3 metres away] ← GREEN box │
│  └──────────────────────────┘                                 │
│                                                               │
│ Scene: You are indoors in a living room, a person is nearby. │
└───────────────────────────────────────────────────────────────┘
```

---

## Other Run Options

```bash
# Use a different camera (e.g., external USB camera)
python demo/main_demo.py --source 1

# Test with a video file (no camera needed)
python demo/main_demo.py --source path/to/video.mp4

# Headless (voice only, no window)
python demo/main_demo.py --no-window

# Silent (print detections, no TTS — for debugging)
python demo/main_demo.py --no-audio
```

---

## After Testing

If you're happy with the output:

1. ✅ Proceed with full training (see `TRAINING_GUIDE.md` in the main project)
2. After training, set in `config.py`:
   ```python
   BLIP_USE_FINETUNED = True
   YOLO_USE_CUSTOM    = True
   ```
3. Run the production app: `python main.py`

If you want to change something (voice speed, danger zone size, description style):
- Edit `demo/config_demo.py` and re-run. **No retraining needed.**
- When behaviour is perfect, apply same changes to `config.py` in the main project.
