# 🦯 Blind-Project — Complete Project Guide

> **A real-time AI assistive navigation system for blind and visually impaired students navigating a university or college campus. The system sees through the camera, understands the environment, and speaks everything in Telugu.**

---

## 📌 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [How Does It Work? (Simple Explanation)](#2-how-does-it-work-simple-explanation)
3. [Full System Architecture](#3-full-system-architecture)
4. [What Can It Detect?](#4-what-can-it-detect)
5. [Datasets Used — Explained](#5-datasets-used--explained)
6. [How the AI Models Are Trained](#6-how-the-ai-models-are-trained)
7. [Expected Accuracy After Training](#7-expected-accuracy-after-training)
8. [Telugu Language Output — How It Works](#8-telugu-language-output--how-it-works)
9. [Technology Stack](#9-technology-stack)
10. [Hardware Requirements](#10-hardware-requirements)
11. [Project File Structure](#11-project-file-structure)
12. [How to Run (Step-by-Step)](#12-how-to-run-step-by-step)
13. [Training Time Estimates](#13-training-time-estimates)
14. [Performance Benchmarks](#14-performance-benchmarks)

---

## 1. What Is This Project?

Imagine you are blind and you want to walk from the college gate to your classroom. You do not know:
- Is there a chair blocking the corridor?
- Is there a staircase ahead?
- What does this sign on the door say?

**This application solves that.** You wear earphones connected to a laptop or mobile device with a camera. The app continuously:

1. 📷 **Watches** your surroundings through the camera
2. 🧠 **Understands** what objects are present and how close they are
3. 🔊 **Speaks** warnings and descriptions in **Telugu** into your earphones

Example output (spoken in Telugu):
> *"హెచ్చరిక! మీకు నేరుగా ముందు ఒక కుర్చీ చాలా దగ్గరగా ఉంది. కుడి వైపు 3 గంటల దిశలో ఒక తలుపు ఉంది."*
> *("Warning! A chair is very close directly ahead. A door is on your right at 3 o'clock.")*

---

## 2. How Does It Work? (Simple Explanation)

Think of it as three jobs happening simultaneously every second:

```
┌─────────────────────────────────────────────────────────┐
│  Job 1: The WATCHER  (YOLO Object Detector)             │
│  → Looks at every camera frame                          │
│  → Draws invisible boxes around objects it recognizes   │
│  → Says: "Chair, 40cm ahead, DANGER ZONE"               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Job 2: The DESCRIBER  (BLIP Caption Model)             │
│  → Every 4 seconds, analyzes the whole scene            │
│  → Says: "You are in a corridor with chairs and a door" │
│  → Combines with YOLO positions for full description    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Job 3: The SPEAKER  (Telugu TTS)                       │
│  → Translates all text to Telugu                        │
│  → Speaks through Microsoft Neural Voice                │
│  → Danger alerts always interrupt description           │
└─────────────────────────────────────────────────────────┘
```

There is also a 4th optional job:

```
┌─────────────────────────────────────────────────────────┐
│  Job 4: The READER  (OCR — press R key)                 │
│  → Reads any text visible in camera (signs, boards)     │
│  → Speaks what it reads in Telugu                       │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Full System Architecture

```
                      ┌──────────────┐
                      │  USB/Built-in│
                      │    Camera    │
                      └──────┬───────┘
                             │ 30 FPS video stream
                             ▼
               ┌─────────────────────────────┐
               │      Vision Module          │
               │   (src/vision_module.py)    │
               │                             │
               │  ┌──────────────────────┐   │
               │  │ CameraStream Thread  │   │
               │  │ (OpenCV VideoCapture)│   │
               │  └──────────┬───────────┘   │
               │             │               │
               │  ┌──────────▼───────────┐   │
               │  │ ObjectDetector       │   │
               │  │ (YOLOv11s inference) │   │
               │  │ ~20-30ms per frame   │   │
               │  └──────────┬───────────┘   │
               │             │               │
               │  ┌──────────▼───────────┐   │
               │  │ SpatialAnalyzer      │   │
               │  │ clock positions      │   │
               │  │ distance estimation  │   │
               │  │ danger zone check    │   │
               │  └──────────┬───────────┘   │
               └─────────────┼───────────────┘
                             │ detections list
              ┌──────────────┼──────────────────┐
              │              │                  │
              ▼              ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │Caption Module│  │  OCR Module  │  │  Danger Zone │
   │(every 4 sec) │  │ (on R press) │  │   ALERT      │
   │              │  │              │  │  (immediate) │
   │ BLIP model   │  │  EasyOCR     │  │              │
   │  Telugu      │  │  en + te     │  │ HIGH priority│
   │  captions    │  │              │  │              │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                  │
          └─────────────────┴──────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │      Audio Module       │
              │  (src/audio_module.py)  │
              │                         │
              │  TeluguTranslator       │
              │  (deep-translator)      │
              │  English → Telugu       │
              │                         │
              │  PriorityTTSQueue       │
              │  HIGH: danger alerts    │
              │  LOW:  descriptions     │
              │                         │
              │  EdgeTTSBackend         │
              │  te-IN-ShrutiNeural     │
              └────────────┬────────────┘
                           │
                           ▼
                    🔊 Speaker / Earphones
                    (Telugu voice output)
```

---

## 4. What Can It Detect?

After training with all datasets, the model will detect the following objects:

### 🏫 Campus / Indoor Objects (from Kaggle + Roboflow datasets)

| Object          | Importance for Blind User            |
| --------------- | ------------------------------------ |
| Door            | Entry/exit — most critical           |
| Open door       | Distinguishes open from closed doors |
| Stairs          | Fall prevention — highest danger     |
| Window          | Orientation, avoid collision         |
| Pole / pillar   | Collision hazard in corridors        |
| Cabinet         | Protruding obstacle                  |
| Ramp            | Alternative to stairs                |

### 👥 People & Vehicles (from COCO)

| Object       | Importance                              |
| ------------ | --------------------------------------- |
| Person       | Most common moving obstacle             |
| Bicycle      | Fast-moving campus hazard               |
| Motorcycle   | Road hazard near campus entrance        |
| Car, Bus     | Parking lot / road hazards              |

### 🪑 Furniture (from COCO + Indoor datasets)

| Object       | Campus Location                 |
| ------------ | ------------------------------- |
| Chair        | Classrooms, corridors, canteen  |
| Couch/sofa   | Common rooms                    |
| Dining table | Canteen, study areas            |
| Bed          | Hostel areas                    |

### 🚦 Signage & Navigation (from COCO)

| Object        | Purpose                       |
| ------------- | ----------------------------- |
| Traffic light | Campus road crossings         |
| Stop sign     | Road safety                   |
| Bench         | Rest areas, gardens           |
| Fire hydrant  | Campus safety awareness       |

### 📦 General Objects (from COCO — 80 classes)

Bottle, cup, backpack, umbrella, handbag, suitcase, book, laptop, phone,
clock, vase, potted plant, cat, dog, and 60+ more standard objects.

**Total after training: ~89 classes** (80 COCO + 9 campus-specific)

---

## 5. Datasets Used — Explained

### What is a "Dataset"?

A dataset is a large collection of images that already have labels attached. For example, an image of a corridor with a box drawn around the chair and the label "chair". The AI learns from thousands of these examples.

---

### Dataset 1 — MS-COCO 2017 (Object Detection + Captions)

| Property      | Details                                  |
| ------------- | ---------------------------------------- |
| Full Name     | Microsoft Common Objects in Context      |
| Size          | 118,000 training images                  |
| Labels        | 80 object categories, millions of boxes  |
| Captions      | 591,753 English scene descriptions        |
| Why we use it | The world's gold standard training set. Teaches the model broad object knowledge with perfect English grammar in captions. |
| Auto-download | ✅ `python data/download_datasets.py --dataset coco` |

---

### Dataset 2 — VizWiz-Captions (Real Blind User Photos)

| Property      | Details                                  |
| ------------- | ---------------------------------------- |
| Full Name     | VizWiz Caption Dataset                   |
| Size          | 23,431 training images                   |
| Who took them | Real blind people using their phones     |
| Captions      | 5 crowd-sourced English captions per image |
| Why we use it | These photos are blurry, tilted, poorly lit — exactly like what our camera will see. The model learns to describe *real* blind-user scenarios, not perfect studio photos. |
| Weight in training | 25% of BLIP training batches       |
| Auto-download | ✅ `python data/download_datasets.py --dataset vizwiz` |

---

### Dataset 3 — AI4Bharat IndicCaption (Telugu Captions) ⭐ PRIMARY

| Property      | Details                                       |
| ------------- | --------------------------------------------- |
| Full Name     | AI4Bharat IndicCOCO — Telugu Subset           |
| Source        | HuggingFace: `ai4bharat/IndicCOCO`            |
| Size          | ~40,000 COCO images with Telugu captions      |
| Language      | **Telugu (తెలుగు)**                           |
| Created by    | IIT Madras AI4Bharat research team           |
| License       | CC-BY-4.0 (free to use)                       |
| Why we use it | This is the **most important dataset** for our goal. It teaches BLIP to write scene descriptions directly in Telugu — no translation needed. The captions describe the same COCO images but written by native Telugu speakers in natural language. |
| Weight in training | **60% of BLIP training batches (highest priority)** |
| Auto-download | ✅ `python data/download_datasets.py --dataset indic` |

---

### Dataset 4, 5, 6 — Indoor Campus Datasets (Manual Download)

| Dataset                  | Platform  | Classes                          | Size     |
| ------------------------ | --------- | -------------------------------- | -------- |
| Indoor Objects Detection | Kaggle    | door, chair, table, window, pole | ~1 GB    |
| Door+Stairs+Chairs       | Roboflow  | door, stairs, chair, toilet      | ~500 MB  |
| SmartCane Indoor Objects | Roboflow  | chair, table, door               | ~300 MB  |

These must be downloaded manually. See [`data/MANUAL_DOWNLOADS.md`](data/MANUAL_DOWNLOADS.md) for the exact steps.

---

## 6. How the AI Models Are Trained

There are two separate AI models in this project. They are trained independently.

---

### Model 1 — YOLO Object Detector

**What it learns:** To recognize and locate objects in a camera image by drawing a bounding box and assigning a class label.

**How it works (simplified):**
```
Input image (640×640 pixels)
    ↓
Divided into a grid of cells
    ↓
Each cell predicts: "Is there an object here? What is it? How confident?"
    ↓
Overlapping boxes are merged (Non-Maximum Suppression)
    ↓
Output: [ {label: "chair", confidence: 0.87, box: (x1,y1,x2,y2)}, ... ]
```

**Training details:**

| Parameter          | Value            | Why                                    |
| ------------------ | ---------------- | -------------------------------------- |
| Base model         | `yolo11s.pt`     | Small — best balance of speed+accuracy |
| Total epochs       | 80               | More epochs = better learning          |
| Image size         | 640 × 640        | Standard for real-time detection       |
| Batch size         | 16               | As many as 4GB VRAM allows             |
| Training data      | COCO + Campus    | Broad + campus-specific objects        |
| Learning rate      | 0.01             | Standard for fine-tuning               |
| Early stopping     | 15 epochs        | Stops if no improvement                |

---

### Model 2 — BLIP Image Captioner

**What it learns:** Given an image, write a sentence describing the scene — in Telugu.

**How it works (simplified):**
```
Input image
    ↓
Vision Encoder (ViT — extracts image features into numbers)
    ↓
Language Decoder (GPT-style — generates Telugu sentence word by word)
    ↓
Output: "మీరు ఒక వసతిగృహ గదిలో ఉన్నారు, మీకు ఎడమ వైపు ఒక మంచం ఉంది."
        ("You are in a hostel room, there is a bed to your left.")
```

**Training details:**

| Parameter            | Value                         | Why                                     |
| -------------------- | ----------------------------- | --------------------------------------- |
| Base model           | `Salesforce/blip-image-captioning-base` | Pre-trained on 129M image-text pairs |
| Epochs               | 8                             | Sufficient for fine-tuning              |
| Batch size           | 4 (effective: 16 via accumulation) | Fits 4GB VRAM                      |
| Learning rate        | 5e-5                          | Standard transformer fine-tuning LR     |
| Telugu data weight   | 60%                           | Primary output language                 |
| VizWiz weight        | 25%                           | Real blind-user context                 |
| COCO weight          | 15%                           | Grammar quality                         |
| VRAM optimization    | Gradient checkpointing + FP16 | Fits RTX 3050 4GB                       |

**Dataset mixing visualization:**
```
Each training batch of 16 images:
  ████████████  10 images from IndicCaption (Telugu)  ← 60%
  █████          4 images from VizWiz (English)        ← 25%
  ███            2 images from COCO (English)           ← 15%
```

---

## 7. Expected Accuracy After Training

### YOLO Detection Accuracy

Accuracy is measured using **mAP (mean Average Precision)** — the higher the better.

| Model Variant | mAP@50 (COCO) | FPS (RTX 3050) | Recommended |
| ------------- | ------------- | -------------- | ----------- |
| yolo11n (nano)| ~37%          | ~45 FPS        | Demo only   |
| **yolo11s (small)** | **~48%** | **~35 FPS** | **✅ Final app** |
| yolo11m (medium) | ~56%       | ~20 FPS        | If 6GB+ VRAM|

After fine-tuning on campus data, accuracy on indoor/campus objects improves by **~8–15%** above base COCO numbers.

### BLIP Caption Quality

Measured using **BLEU** and **METEOR** scores. These measure how similar the generated sentence is to a human-written reference sentence.

| Metric  | Score Range | Our Target | What It Means                        |
| ------- | ----------- | ---------- | ------------------------------------ |
| BLEU-1  | 0.0 – 1.0   | ≥ 0.55     | At least 55% of individual words match |
| BLEU-4  | 0.0 – 1.0   | ≥ 0.25     | 4-word phrases match reference       |
| METEOR  | 0.0 – 1.0   | ≥ 0.28     | Word match including synonyms         |

> **Note:** Scores appear low but are standard for image captioning. The practical output quality is much better than numbers suggest — especially for navigation, where the direction and proximity are more important than exact word matching.

### Telugu Output Quality

| Before Training (Demo)     | After Training (Final App)         |
| -------------------------- | ---------------------------------- |
| Gemini describes in English | BLIP describes scene natively       |
| deep-translator translates  | **No translation needed**           |
| ~85% accurate Telugu       | **~95%+ natural Telugu**           |
| API call needed (internet)  | Runs fully locally                 |
| ~2 second delay             | **< 1 second latency**             |

---

## 8. Telugu Language Output — How It Works

### Phase 1: Current Demo (Translation-based)

```
English description → deep-translator → Telugu text → edge-tts → 🔊 Telugu voice
```

- Works immediately, no training needed
- Requires internet for translation + TTS
- Accuracy: ~85–90% natural Telugu

### Phase 2: After Training (Native Telugu)

```
Camera image → Fine-tuned BLIP → Telugu text (direct) → edge-tts → 🔊 Telugu voice
```

- BLIP generates Telugu directly — **no translation step**
- Faster, more accurate, more natural
- TTS still needs internet (Microsoft neural voice)

### Telugu Voice Settings

| Setting        | Value                  | Description                    |
| -------------- | ---------------------- | ------------------------------ |
| `TTS_ENGINE`   | `edge-tts`             | Microsoft neural TTS           |
| `EDGE_TTS_VOICE` | `te-IN-ShrutiNeural` | Telugu female neural voice     |
| `TELUGU_MODE`  | `True`                 | Enables translation pipeline   |
| Switch to male | `te-IN-MohanNeural`    | Change in `config.py`          |
| Switch to English | Set `TELUGU_MODE = False` | English mode         |

---

## 9. Technology Stack

| Category           | Technology                              | Purpose                                  |
| ------------------ | --------------------------------------- | ---------------------------------------- |
| Language           | Python 3.10+                            | Core programming language                |
| Computer Vision    | OpenCV 4.9+                             | Camera feed, frame capture               |
| Object Detection   | Ultralytics YOLOv11s                    | Real-time object detection               |
| Image Captioning   | Salesforce BLIP (HuggingFace)           | Scene description generation             |
| Deep Learning      | PyTorch 2.1+ + CUDA                     | Model training and inference             |
| NLP / Transformers | HuggingFace Transformers                | BLIP model loading and training          |
| OCR                | EasyOCR 1.7+                            | Reading text from camera (signs, boards) |
| Translation        | deep-translator 1.11+                   | English → Telugu translation             |
| Text-to-Speech     | Microsoft edge-tts                      | Telugu neural voice output               |
| Audio Playback     | pygame                                  | MP3 audio playback for edge-tts          |
| Optimization       | ONNX + OpenVINO                         | Export for CPU/edge deployment           |
| Concurrency        | Python ThreadPoolExecutor               | Parallel vision + caption + TTS          |
| Telugu Dataset     | AI4Bharat IndicCOCO (HuggingFace)       | Telugu caption training data             |
| General Dataset    | MS-COCO 2017 + VizWiz                   | Object detection + English captions      |

---

## 10. Hardware Requirements

| Component   | Minimum (Training)   | Minimum (Inference only) | Recommended         |
| ----------- | -------------------- | ------------------------ | ------------------- |
| GPU         | NVIDIA 4 GB VRAM     | No GPU needed (slow)     | NVIDIA 6 GB+ VRAM   |
| RAM         | 16 GB                | 8 GB                     | 32 GB               |
| Disk Space  | 60 GB free           | 5 GB (models only)       | 100 GB free         |
| Python      | 3.10+                | 3.10+                    | 3.10+               |
| OS          | Windows 10/11        | Windows / Linux          | Windows 10/11       |
| Internet    | Required (downloads) | Required (TTS only)      | Required            |

---

## 11. Project File Structure

```
Blind-Project/
│
├── config.py                   ← Central settings for EVERYTHING
├── main.py                     ← Run this for the full application
├── requirements.txt            ← Install all Python libraries
│
├── src/                        ← Application core
│   ├── vision_module.py        ← Camera + YOLO detection + spatial reasoning
│   ├── caption_module.py       ← BLIP scene description
│   ├── audio_module.py         ← Telugu TTS (translation + voice)
│   └── ocr_module.py           ← Sign/text reading mode
│
├── data/                       ← All training data lives here
│   ├── download_datasets.py    ← Auto-download script
│   ├── dataset_loader.py       ← PyTorch Dataset classes
│   ├── augmentations.py        ← Image augmentation (flip, brightness, etc.)
│   ├── MANUAL_DOWNLOADS.md     ← Guide for Kaggle/Roboflow downloads
│   ├── vizwiz/                 ← VizWiz images + annotations (auto)
│   ├── coco/                   ← COCO images + captions (auto)
│   ├── indic_caption/          ← Telugu IndicCaption JSON (auto)
│   └── indoor_campus/          ← Campus datasets (manual download)
│       ├── indoor_objects/     ← From Kaggle
│       ├── door_stairs/        ← From Roboflow
│       └── smartcane/          ← From Roboflow
│
├── training/                   ← All training scripts
│   ├── train_detector.py       ← YOLO training (COCO + campus combined)
│   ├── train_captioner.py      ← BLIP training (Telugu-primary)
│   ├── evaluate.py             ← Quality measurement (BLEU/METEOR)
│   └── export_models.py        ← Save as ONNX/OpenVINO
│
├── checkpoints/                ← Saved AI weights (created after training)
│   ├── yolo11_custom.pt        ← Trained YOLO weights
│   └── blip_finetuned/
│       ├── checkpoint_epoch1/  ← Saved every epoch
│       └── best/               ← Best performing checkpoint
│
├── logs/                       ← Training logs and evaluation reports
├── tests/                      ← Unit tests
│
├── demo/                       ← Demo version (no training required)
│   ├── main_demo.py            ← Entry point
│   └── config_demo.py          ← Demo-specific settings
│
├── IMPLEMENTATION_PLAN.md      ← Technical design document
├── TRAINING_GUIDE.md           ← Step-by-step training instructions
├── NEXT_STEPS_DATASET_UPGRADE.md ← Dataset download checklist
└── data/MANUAL_DOWNLOADS.md   ← Kaggle/Roboflow download guide
```

---

## 12. How to Run (Step-by-Step)

### Option A — Run the Demo Right Now (No Training)

```bash
# Install demo dependencies
pip install -r demo/requirements_demo.txt

# Run demo (uses Gemini AI + pre-trained YOLO + Telugu voice)
python demo/main_demo.py
```

### Option B — Full Training + Production Run

```bash
# Step 1: Install all dependencies
pip install -r requirements.txt
pip install datasets huggingface-hub deep-translator edge-tts pygame

# Step 2: Auto-download datasets
python data/download_datasets.py --dataset all

# Step 3: Download indoor campus datasets manually
#         Open data/MANUAL_DOWNLOADS.md and follow the steps

# Step 4: Train YOLO (campus object detector)
python training/train_detector.py --dataset combined --model yolo11s.pt

# Step 5: Train BLIP (Telugu scene captioner)
python training/train_captioner.py

# Step 6: Evaluate quality
python training/evaluate.py

# Step 7: Enable trained models
#         Open config.py and set:
#         YOLO_USE_CUSTOM   = True
#         BLIP_USE_FINETUNED = True

# Step 8: Run the full application
python main.py
```

### Keyboard Controls (while running)

| Key     | Action                         |
| ------- | ------------------------------ |
| `Q`     | Quit the application           |
| `R`     | Toggle reading mode (OCR)      |
| `P`     | Pause / resume Telugu voice    |
| `Ctrl+C`| Graceful shutdown              |

---

## 13. Training Time Estimates

Based on **NVIDIA RTX 3050 (4 GB VRAM)**:

| Task                                      | Estimated Time      |
| ----------------------------------------- | ------------------- |
| Install Python libraries                  | 20 – 40 minutes     |
| Download VizWiz (~5 GB)                   | 30 – 60 minutes     |
| Download MS-COCO (~20 GB)                 | 1.5 – 3 hours       |
| Download AI4Bharat IndicCaption           | 10 – 20 minutes     |
| Download indoor campus datasets (manual)  | 30 – 60 minutes     |
| **Total download time**                   | **~3 – 5 hours**    |
| YOLO fine-tuning (80 epochs)              | 2 – 4 hours         |
| BLIP fine-tuning (8 epochs)               | 4 – 7 hours         |
| Evaluation + export                       | ~1 hour             |
| **Total training time**                   | **~7 – 12 hours**   |
| **Grand total (first-time setup)**        | **~10 – 17 hours**  |

> 💡 After this one-time training, the app runs instantly every time. Training is done only once.

---

## 14. Performance Benchmarks

### Real-Time Inference (RTX 3050)

| Component                    | Latency         | Notes                            |
| ---------------------------- | --------------- | -------------------------------- |
| YOLO object detection        | 20 – 35 ms      | ~30 FPS real-time                |
| BLIP scene caption (trained) | 200 – 400 ms    | Runs every 4 sec in background   |
| Telugu translation           | 100 – 300 ms    | Only in demo mode (pre-training) |
| edge-tts voice generation    | 800 – 1500 ms   | Depends on internet speed        |
| **Total glass-to-ear**       | **< 2 seconds** | From camera to spoken word       |
| Danger alert response        | **< 200 ms**    | Instant interrupt, HIGH priority |

### Success Metrics

| Metric                     | Target Value   | Meaning                                          |
| -------------------------- | -------------- | ------------------------------------------------ |
| Detection FPS              | ≥ 25 FPS       | Smooth, real-time detection                      |
| BLEU-4 score               | ≥ 0.25         | Caption sentence quality                         |
| METEOR score               | ≥ 0.28         | Synonym-aware caption quality                    |
| Danger alert latency       | < 200 ms       | Obstacle warning reaches user almost instantly   |
| Glass-to-ear total latency | < 2 seconds    | Time from seeing object to speaking in Telugu    |
| Telugu voice clarity       | Natural speech | te-IN-ShrutiNeural Microsoft neural voice        |

---

*Built for visually impaired students — designed to make every campus navigable.*
*Last updated: 2026-03-12*
