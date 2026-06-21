# Blind-Project Comprehensive Knowledge Base

> **Updated: June 2026** — Reflects current mBLIP architecture, FastAPI mobile deployment, and full project state.

This document is the master reference for the **Blind-Project**: an assistive navigation system for blind users navigating a college or university campus. It covers architecture, tech stack, training workflow, deployment options, and spatial logic.

---

## 1. Project Overview & Goal

**Goal:** Build a real-time, low-latency assistive navigation application that detects indoor campus objects and hazards, generating spoken Telugu descriptions and alerts to guide visually impaired users safely.

**Key Features:**
- **Real-Time Obstacle Detection:** Detects 18 campus-specific objects — doors, stairs, poles, tables, chairs, people, bicycles, ramps, etc.
- **Danger Zone Analysis:** Spatial reasoning determines if an object is "very close", "nearby", or "in the distance", and whether it lies in the critical centre collision path (Danger Zone: 28% × 40% of frame).
- **Native Telugu Output:** Scene descriptions and urgent hazard alerts are generated directly in Telugu and spoken via Microsoft Neural TTS (`te-IN-ShrutiNeural`).
- **Two Deployment Modes:** Desktop OpenCV developer window (`main.py`) or mobile client-server architecture (`api.py` + `frontend/index.html`).
- **Auto OCR:** Automatically reads text from signs and notice boards when YOLO detects them (no user action needed).

---

## 2. Tech Stack & Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Object Detection | YOLO11s (Ultralytics) | 18 campus-class real-time inference |
| Image Captioning (Production) | mBLIP `Gregor/mblip-mt0-xl` + LoRA | Native Telugu scene descriptions |
| Image Captioning (Demo) | Google Gemini 1.5 Flash API | Demo-only, no local GPU needed |
| Captioning Quantization | bitsandbytes (4-bit NF4) | Fits mBLIP in 4 GB VRAM |
| Captioning Adapter | PEFT LoRA | Campus fine-tuning (~100 MB adapter) |
| Backend Framework | FastAPI + uvicorn | Mobile deployment API server |
| Frontend App | Pure HTML5/JavaScript PWA | Mobile camera + Web Speech API |
| Computer Vision | OpenCV (cv2) | Frame capture, resize, scene change detection |
| TTS Engine (Primary) | Microsoft edge-tts `te-IN-ShrutiNeural` | Telugu neural voice (internet required) |
| TTS Engine (Fallback) | pyttsx3 | Offline English-only fallback |
| Translation | deep-translator (GoogleTranslator) | English alert strings → Telugu |
| OCR | EasyOCR (Telugu + English) | Sign/notice/board reading |
| Camera Capture | OpenCV VideoCapture (threaded) | Ring buffer frame capture |
| Evaluation | NLTK (BLEU + METEOR) | Caption quality metrics |
| Image Augmentation | torchvision + PIL | Training data augmentation pipeline |

---

## 3. Project Architecture & Flow

### Desktop Mode (main.py)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py (BlindNavigationApp)                  │
│                                                                       │
│  Thread: CameraStream                Thread: TTSWorkerThread          │
│  ┌─────────────────────┐            ┌──────────────────────────────┐ │
│  │ cv2.VideoCapture()  │            │ PriorityTTSQueue              │ │
│  │ ring buffer (5 fr)  │            │  HIGH → tts.alert() (clears) │ │
│  └──────┬──────────────┘            │  LOW  → tts.speak() (queue)  │ │
│         │ frame                     └──────────────────────────────┘ │
│  Main loop (main thread):                                             │
│   ├─► YOLO detect (sync, ~30ms)                                       │
│   │     └─► SpatialAnalyzer → clock_pos, dist, danger zone           │
│   ├─► HIGH-PRIORITY alert? → tts.alert() (immediate)                 │
│   ├─► DANGER ZONE alert?   → tts.alert()                             │
│   ├─► Submit captioner to ThreadPoolExecutor (async, 4s interval)    │
│   │     └─► mBLIP → Telugu caption → tts.speak()                    │
│   ├─► OCR auto-trigger (if sign/board detected by YOLO)              │
│   │     └─► EasyOCR → text → tts.alert()                            │
│   └─► draw_dev_overlay() → cv2.imshow()                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Mobile API Mode (api.py + frontend/index.html)

```
Mobile Phone (Browser)               PC Server (FastAPI)
┌───────────────────────┐           ┌──────────────────────────────────┐
│ navigator.mediaDevices │           │ POST /analyze                    │
│ getUserMedia()         │  base64  │  ├─► decode_base64_image()       │
│ canvas.toDataURL()     │─────────►│  ├─► YOLO detect (sync, ~30ms)  │
│ fetch('/analyze', POST)│           │  ├─► build_description()         │
│                        │           │  ├─► build_danger_alert()        │
│ handleResponse():      │◄─────────│  └─► BackgroundTask: mBLIP       │
│  alerts → cancel+speak │  JSON    │       (async, every 4s)           │
│  caption → speak       │           └──────────────────────────────────┘
│  speechSynthesis te-IN │
└───────────────────────┘
```

---

## 4. Demonstration Version (Gemini API Demo)

The `demo/` directory is a **standalone** demo for presentations without GPU training.

### Why a Demo Version?
Training the full mBLIP model requires:
- 4 GB+ GPU (RTX 3050 or better)
- 300–500 campus photos with Telugu captions (custom collected)
- ~30 minutes of LoRA fine-tuning

The demo bypasses all of this using Google Gemini Flash API for captioning.

### Demo Architecture
- **YOLO:** Standard `yolo11n.pt` (Nano, pre-trained on COCO) — no campus classes
- **Captioning:** `demo/caption_module_demo.py` → Gemini 1.5 Flash API (via `google-generativeai`)
- **Prompt:** `"Describe what you see in this image in ONE clear, concise sentence in TELUGU... Do NOT say 'The image shows'."`
- **Rate limit:** 4.0s interval between API calls (Gemini free tier: 15 RPM)
- **Everything else:** SpatialReasoningNLP, TTSWorker, OCRReader, draw_dev_overlay — **identical** to production

### How to Run the Demo
1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Set it in `demo/config_demo.py`: `GEMINI_API_KEY = "your_key_here"`
3. Install: `pip install -r demo/requirements_demo.txt`
4. Run: `python demo/main_demo.py`

### Demo vs Production Comparison

| Component | Demo (`demo/`) | Production (`main.py`) |
|-----------|---------------|------------------------|
| Captioning | Gemini Flash API (internet) | mBLIP LoRA fine-tuned (local GPU) |
| YOLO | yolo11n.pt (COCO 80 classes) | yolo11s.pt → campus fine-tuned (18 classes) |
| TTS, OCR, Spatial | Identical | Identical |
| Requires internet | Yes (Gemini API + edge-tts) | edge-tts only |
| Requires GPU | No | 4 GB VRAM minimum |

---

## 5. Production Version (mBLIP + YOLO11s)

### YOLO11s Campus Classes (18 total)

Trained on campus-specific Roboflow datasets. Excludes irrelevant COCO classes (giraffes, frisbees, pizza) that cause class confusion indoors.

```python
CAMPUS_CLASSES = [
    "person", "bicycle", "motorcycle", "car",
    "bench", "chair", "table", "backpack", "laptop", "cell phone",
    "door", "openedDoor", "window", "stairs", "step", "ramp",
    "pole", "corridor",
]
HIGH_PRIORITY_OBJECTS = {"stairs", "step", "ramp", "openedDoor", "pole"}
```

### mBLIP Configuration

```python
MBLIP_PRETRAINED_NAME  = "Gregor/mblip-mt0-xl"     # ~5 GB download, once
MBLIP_FINETUNED_PATH   = "checkpoints/mblip_campus" # LoRA adapter save path
MBLIP_USE_FINETUNED    = False    # Set True after training
MBLIP_USE_4BIT         = True     # 4-bit NF4 for RTX 3050 4 GB
MBLIP_PROMPT           = "క్లుప్తంగా వివరించు:"  # Telugu: "Describe briefly"
MBLIP_MAX_NEW_TOKENS   = 25       # Short navigation captions
MBLIP_NUM_BEAMS        = 1        # Greedy (fastest)
MBLIP_CAPTION_INTERVAL = 4.0      # Seconds between caption calls
SCENE_CHANGE_THRESHOLD = 15.0     # Skip if avg pixel diff < this
```

### Inference Speed Optimizations

1. **4-bit NF4 quantization** — bitsandbytes, VRAM: ~3.5 GB (vs 8 GB float16)
2. **Greedy search** — num_beams=1, no sampling
3. **25 token cap** — Navigation captions are short
4. **224×224 resize** — Before Q-Former encoding (from 384×384)
5. **Scene change detection** — cv2.absdiff on grayscale, skip if mean diff < 15.0

---

## 6. Training & Data Collection Workflow

### Dataset Sources

| Dataset | Purpose | Source |
|---------|---------|--------|
| Your campus photos + Telugu captions | mBLIP LoRA fine-tuning | You collect (300–500 photos) |
| SCIN Indoor Navigation (544 images) | YOLO training | Roboflow |
| Akhash Indoor Navigation (1,115 images) | YOLO training | Roboflow |
| Blind Indoor Navigation | YOLO training | Roboflow |
| Stairs Detection (1,000 images) | YOLO training | Kaggle (Samuel Ayman) |

> **All previously referenced HuggingFace Telugu image-caption datasets are deleted/private** (as of March 2026). See `IMPLEMENTATION_PLAN.md` for full investigation.

### Training Commands

```bash
# 1. Prepare datasets
python data/download_datasets.py --dataset campus-setup   # Create campus caption folder
python data/download_datasets.py --dataset verify          # Check readiness

# 2. Train YOLO campus detector
python training/train_detector.py --dataset campus

# 3. Fine-tune mBLIP with LoRA
python training/train_captioner.py

# 4. Evaluate
python training/evaluate.py

# 5. Activate in config.py
#    YOLO_USE_CUSTOM = True
#    MBLIP_USE_FINETUNED = True

# 6. Run
python main.py
```

Full guide: **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

---

## 7. How to Run — All Modes

### Desktop Developer Mode

```bash
python main.py                      # Live webcam (camera 0)
python main.py --source 1           # External camera
python main.py --source video.mp4   # Test video
python main.py --no-window          # Headless (no OpenCV window)
python main.py --no-audio           # Text output only (debugging)
```

### FastAPI Mobile Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000

# Serve frontend:
cd frontend && python -m http.server 8080
# Open on phone: http://<PC_IP>:8080
```

### Demo Mode (No Training Required)

```bash
pip install -r demo/requirements_demo.txt
python demo/main_demo.py
```

---

## 8. Detailed Spatial Logic

### Clock Position Grid (SpatialAnalyzer.clock_position)

The frame is divided into a 3×3 grid. The bounding box centre determines clock position:

```
Frame (640 × 480 pixels):
┌──────────┬──────────┬──────────┐
│ 10 o'clk │ 12 o'clk │  2 o'clk │  (top row, ny < 0.33)
├──────────┼──────────┼──────────┤
│  9 o'clk │  directly│  3 o'clk │  (middle row, 0.33 ≤ ny < 0.67)
│          │   ahead  │          │
├──────────┼──────────┼──────────┤
│  8 o'clk │  6 o'clk │  4 o'clk │  (bottom row, ny ≥ 0.67)
└──────────┴──────────┴──────────┘
     left      centre      right
  (nx < 0.33) (0.33-0.67) (nx > 0.67)
```

### Distance Estimation (SpatialAnalyzer.distance_word)

```python
bbox_area / frame_area:
  ≥ 15% → "very close"    (DISTANCE_NEAR_THRESHOLD   = 0.15)
  ≥  3% → "nearby"        (DISTANCE_MEDIUM_THRESHOLD = 0.03)
  <  3% → "in the distance"
```

### Danger Zone (SpatialAnalyzer.in_danger_zone)

```
Danger zone = centre 28% (width) × 40% (height) of frame:
  X: [frame_w × 0.36, frame_w × 0.64]  (28% centred)
  Y: [frame_h × 0.30, frame_h × 0.70]  (40% centred)

Any bbox overlapping this rectangle → in_danger_zone = True
```

### Alert Priority Chain

```
Detection → is_high_priority?
    YES → tts.alert()  # HIGH priority — clears queue, immediate
    NO  → in_danger_zone?
              YES → tts.alert()  # Danger warning
              NO  → included in scene description → tts.speak()  # LOW priority
```

### SpatialReasoningNLP.build_description Output Example

```
"మీరు నడవలో ఉన్నారు. హెచ్చరిక: stairs nearby, directly ahead! 
జాగ్రత్త: chair very close, 9 o'clock. person in the distance, 2 o'clock."
```

(mBLIP Telugu caption first, then high-priority warnings, then danger zone objects, then remaining detections)

---

## 9. OCR System

`OCRReader` in `src/ocr_module.py` wraps EasyOCR with bilingual Telugu + English support.

**Two trigger modes:**
1. **Auto:** YOLO detects a `sign`, `notice`, `board`, `poster`, or `text` class with confidence ≥ 0.60
2. **Manual:** User presses `R` key in developer window

**Rate limiting:**
- OCR inference: maximum once every 2 seconds (`_OCR_INTERVAL = 2.0`)
- Auto-trigger cooldown: 5 seconds between auto-triggers (`_AUTO_COOLDOWN = 5.0`)

**Output format:** `"Sign reads: Text1 | Text2 | Text3"` → sent to `tts.alert()` (HIGH priority)

---

## 10. Audio System

`TTSWorker` in `src/audio_module.py` runs on a dedicated background thread:

```
TTSWorker
├── PriorityTTSQueue (thread-safe)
│   ├── HIGH priority (0) — danger alerts → jumps to front
│   └── LOW priority  (1) — scene descriptions → plays in order
├── EdgeTTSBackend (default)
│   ├── edge_tts.Communicate(text, "te-IN-ShrutiNeural")
│   ├── saves to temp .mp3 → pygame.mixer.music.play()
│   └── falls back to Pyttsx3Backend if edge-tts/pygame unavailable
└── Pyttsx3Backend (fallback)
    └── English only — no Telugu neural voice offline
```

**TeluguTranslator:** English danger alert strings are translated to Telugu via `deep-translator → GoogleTranslator(source="en", target="te")` before speaking. Scene descriptions from mBLIP are already in Telugu — no translation needed.

---

## 11. Configuration Reference (config.py)

### Frequently Changed Settings

```python
# After YOLO training:
YOLO_USE_CUSTOM = True
YOLO_CUSTOM_WEIGHTS = "checkpoints/yolo11_campus.pt"

# After mBLIP LoRA training:
MBLIP_USE_FINETUNED = True
MBLIP_FINETUNED_PATH = "checkpoints/mblip_campus"

# For offline use (no internet):
TTS_ENGINE = "pyttsx3"   # English only fallback
TELUGU_MODE = False

# For cloud training (8+ GB VRAM):
MBLIP_USE_4BIT = False
MBLIP_TRAIN_BATCH_SIZE = 4
MBLIP_GRAD_ACCUM_STEPS = 4

# For higher detection accuracy (8+ GB VRAM):
YOLO_MODEL_NAME = "yolo11m.pt"  # instead of yolo11s.pt
```

### Hardware Settings

```python
CAMERA_INDEX  = 0         # Change for external USB camera
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
```
