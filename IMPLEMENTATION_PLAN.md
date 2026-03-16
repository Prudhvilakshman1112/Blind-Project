# 🦯 Blind-Project: Telugu Campus Navigation Assistant
## Implementation Plan

> **Objective**: An AI-powered assistive navigation application for blind and visually impaired students inside university/college premises. The app detects surrounding objects, understands the scene, reads signs, and speaks everything in **Telugu** via a neural voice.

---

## System Architecture

```
Camera Feed (OpenCV)
        │
        ▼
  Vision Module ──────────────────────────────────────────────┐
  (YOLOv11s — trained on COCO + Indoor Campus datasets)       │
  (Danger Zone Alert — center 35% × 50% of frame)            │
  (Spatial Grid: 9-zone clock positions)                      │
        │                                                     │
        ▼                                                     ▼
  Caption Module                                       OCR Module
  (BLIP — fine-tuned on Telugu IndicCaption 60%)      (EasyOCR — en + te)
  (Spatial Reasoning NLP layer)                               │
        │                                                     │
        └──────────────────┬───────────────────────────────────┘
                           ▼
                    Audio Module
              (Telugu Translation — deep-translator)
              (Microsoft Neural TTS — te-IN-ShrutiNeural)
              (Priority Queue: Danger > Description > OCR)
                           │
                           ▼
                    Speaker / Earphones
                    🔊 "హెచ్చరిక! మీకు ముందు ఒక కుర్చీ ఉంది"
```

---

## Dataset Strategy

### Phase 1 — Object Detection (YOLO)

| Dataset                        | Source          | Classes                             | Size      | Auto? |
| ------------------------------ | --------------- | ----------------------------------- | --------- | ----- |
| MS-COCO 2017                   | Auto-download   | 80 general objects                  | ~20 GB    | ✅    |
| Indoor Objects Detection       | Kaggle (manual) | door, chair, table, window, pole    | ~1 GB     | ❌    |
| Door + Stairs + Chairs         | Roboflow (manual)| door, stairs, chair, toilet         | ~500 MB   | ❌    |
| SmartCane Indoor Objects       | Roboflow (manual)| chair, table, door                  | ~300 MB   | ❌    |

👉 Manual download guide: `data/MANUAL_DOWNLOADS.md`

### Phase 2 — Scene Captioning (BLIP — Telugu)

| Dataset                  | Source         | Language   | Training Weight | Auto? |
| ------------------------ | -------------- | ---------- | --------------- | ----- |
| AI4Bharat IndicCaption   | HuggingFace    | **Telugu** | **60%**         | ✅    |
| VizWiz-Captions          | Auto-download  | English    | 25%             | ✅    |
| MS-COCO Captions         | Auto-download  | English    | 15%             | ✅    |

---

## Module Descriptions

### `src/vision_module.py`
- **CameraStream**: Threaded OpenCV `VideoCapture` with frame buffer
- **ObjectDetector**: YOLOv11s inference → bounding boxes + class labels + confidence
- **SpatialAnalyzer**:
  - Divides frame into 3×3 grid → clock-position labels (9, 12, 3 o'clock etc.)
  - Bounding box area % → distance estimate (`very close` / `nearby` / `in the distance`)
  - Center 35% × 50% of frame = **danger zone** → triggers priority interrupt alert

### `src/caption_module.py`
- **SceneCaptioner**: Loads fine-tuned BLIP (Telugu-trained)
- **SpatialReasoningNLP**: Enriches BLIP output with YOLO detections
  - e.g. `"You are in a corridor. A chair is very close at 9 o'clock."`
  - Scene description fires every **4 seconds** (rate-limited)

### `src/audio_module.py`
- **TeluguTranslator**: Translates English text → Telugu via `deep-translator`
- **EdgeTTSBackend**: Microsoft neural voice `te-IN-ShrutiNeural`
- **PriorityTTSQueue**: Two-tier queue
  - `HIGH` priority → danger alerts (interrupt immediately)
  - `LOW` priority → scene descriptions + OCR

### `src/ocr_module.py`
- **OCRReader**: EasyOCR — reads English + Telugu text from camera
- Press `R` in developer window to toggle Reading Mode

### `main.py`
- `ThreadPoolExecutor` — parallel capture + inference + TTS
- Developer Window: bounding boxes, clock positions, danger zone, FPS
- Keyboard: `Q` quit · `R` reading mode · `P` pause TTS

---

## File Structure

```
Blind-Project/
├── config.py                       ← All settings (TELUGU_MODE, model paths, etc.)
├── main.py                         ← Production entry point
│
├── src/                            ← Core application modules
│   ├── audio_module.py             ← Telugu TTS + translator
│   ├── caption_module.py           ← BLIP scene captioning
│   ├── vision_module.py            ← YOLO detection + spatial reasoning
│   └── ocr_module.py               ← Sign/text reading
│
├── data/
│   ├── download_datasets.py        ← Auto-download: VizWiz, COCO, IndicCaption
│   ├── dataset_loader.py           ← IndicCaptionDataset + CombinedCaptionDataset
│   ├── MANUAL_DOWNLOADS.md         ← Kaggle/Roboflow step-by-step guide
│   ├── vizwiz/                     ← VizWiz blind-user captions
│   ├── coco/                       ← COCO images + captions
│   ├── indic_caption/              ← AI4Bharat Telugu captions
│   └── indoor_campus/              ← Campus datasets (manual)
│
├── training/
│   ├── train_detector.py           ← YOLO: COCO + campus combined training
│   ├── train_captioner.py          ← BLIP: Telugu-primary fine-tuning
│   ├── evaluate.py                 ← BLEU/METEOR scoring
│   └── export_models.py            ← ONNX + OpenVINO export
│
├── demo/                           ← Standalone demo (no training needed)
│   └── main_demo.py                ← Gemini Flash + pre-trained YOLO + Telugu TTS
│
└── checkpoints/                    ← Saved weights after training
    ├── yolo11_custom.pt
    └── blip_finetuned/best/
```

---

## Hardware Requirements

| Component   | Minimum              | Recommended         |
| ----------- | -------------------- | ------------------- |
| GPU         | NVIDIA 4 GB VRAM     | NVIDIA 6 GB+ VRAM   |
| RAM         | 16 GB                | 32 GB               |
| Disk Space  | 60 GB free           | 100 GB free         |
| Python      | 3.10+                | 3.10+               |
| OS          | Windows 10/11        | Windows 10/11       |

---

## Time Estimates (NVIDIA RTX 3050)

| Phase                              | Estimated Time  |
| ---------------------------------- | --------------- |
| Dataset download (all, ~25 GB)     | 2 – 4 hours     |
| Library installation               | ~30 minutes     |
| BLIP fine-tuning (8 epochs)        | 4 – 7 hours     |
| YOLOv11s fine-tuning (80 epochs)   | 2 – 4 hours     |
| Evaluation + ONNX export           | ~1 hour         |
| **Total one-time setup**           | **~10 – 16 hours** |

---

## Success Metrics

| Metric                        | Target       |
| ----------------------------- | ------------ |
| Real-time FPS                 | ≥ 25 FPS     |
| Glass-to-ear latency          | < 1 second   |
| BLEU-4 (VizWiz val set)       | ≥ 0.25       |
| METEOR (VizWiz val set)       | ≥ 0.28       |
| Danger alert response time    | < 200 ms     |
| Telugu TTS startup            | < 2 seconds  |

---

## Technology Stack

| Category          | Technology                                        |
| ----------------- | ------------------------------------------------- |
| Language          | Python 3.10+                                      |
| Computer Vision   | OpenCV, Ultralytics YOLOv11s                      |
| Image Captioning  | Salesforce BLIP (fine-tuned on Telugu)            |
| Deep Learning     | PyTorch + HuggingFace Transformers                |
| OCR               | EasyOCR (English + Telugu)                        |
| Translation       | deep-translator (Google Translate API)            |
| Speech (TTS)      | Microsoft edge-tts — `te-IN-ShrutiNeural`         |
| Optimization      | ONNX + OpenVINO (for edge deployment)             |
| Concurrency       | `concurrent.futures.ThreadPoolExecutor`           |
| Datasets          | IndicCaption (te) + VizWiz + COCO + Campus sets   |

---

*Last updated: 2026-03-12*
