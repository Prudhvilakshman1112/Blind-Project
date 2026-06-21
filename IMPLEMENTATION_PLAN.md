# Implementation Plan — Blind-Project Architecture & Change History

> **Last Updated: June 2026**
> This document records the complete architecture, all major overhauls, bug fixes, and the current state of the Blind-Project system.

---

## Project Summary

**Blind-Project** is a real-time assistive navigation application for blind/visually impaired users on a college or university campus. It detects objects, describes scenes in Telugu, and speaks alerts via a priority TTS system.

### Two Deployment Modes

1. **Desktop app** (`main.py`) — OpenCV developer window, webcam, local TTS via edge-tts
2. **Mobile server** (`api.py` + `frontend/index.html`) — FastAPI backend, phone as camera, Web Speech API for TTS

---

## Current Architecture (June 2026)

### Module Map

```
Blind-Project/
├── main.py              ← Desktop app controller (ThreadPoolExecutor, 3 concurrent threads)
├── api.py               ← FastAPI server (YOLO sync + mBLIP async background tasks)
├── config.py            ← All constants, model paths, thresholds, TTS settings
│
├── src/
│   ├── vision_module.py   ← CameraStream, SpatialAnalyzer, ObjectDetector (YOLO11)
│   ├── caption_module.py  ← SceneCaptioner (mBLIP), SpatialReasoningNLP
│   ├── audio_module.py    ← TTSWorker, PriorityTTSQueue, EdgeTTSBackend, TeluguTranslator
│   └── ocr_module.py      ← OCRReader (EasyOCR bilingual Te+En)
│
├── data/
│   ├── download_datasets.py   ← Campus caption folder setup + dataset verification
│   ├── dataset_loader.py      ← CampusCaptionDataset, CampusDetectionVerifier, get_dataloaders
│   ├── augmentations.py       ← Image augmentation pipeline (motion blur, brightness, noise)
│   ├── MANUAL_DOWNLOADS.md    ← Roboflow/Kaggle dataset download instructions
│   └── campus_captions/       ← Human-collected campus images + Telugu captions
│       ├── train.json          ← Training annotation JSON (currently: 2 sample entries)
│       ├── val.json            ← Validation annotation JSON (currently: 1 sample entry)
│       └── images/             ← Campus .jpg photos (empty — user fills this)
│
├── training/
│   ├── train_detector.py  ← YOLO11s fine-tuning on campus Roboflow datasets
│   ├── train_captioner.py ← mBLIP LoRA fine-tuning on campus Telugu captions
│   ├── evaluate.py        ← BLEU-1/2/3/4 + METEOR evaluation (campus val split)
│   └── export_models.py   ← ONNX + OpenVINO export (BLIP vision encoder + YOLO)
│
├── frontend/
│   └── index.html         ← PWA mobile client (8 FPS, POST to /analyze, Web Speech API te-IN)
│
├── demo/                  ← Standalone demo (Gemini API + yolo11n.pt, no training needed)
│   ├── main_demo.py
│   ├── api_demo.py
│   ├── config_demo.py
│   ├── caption_module_demo.py  ← Uses Gemini instead of mBLIP
│   ├── vision_module.py
│   ├── audio_module.py
│   ├── ocr_module.py
│   └── README_DEMO.md
│
└── tests/
    ├── test_audio_module.py
    └── test_spatial_analyzer.py
```

---

## Core Technical Decisions

### YOLO11s — 18 Campus Classes Only

**Problem:** Training on full COCO 80 classes (elephants, pizzas, frisbees) wastes GPU time and confuses the model indoors.

**Solution:** Custom 18-class campus model:
```python
CAMPUS_CLASSES = [
    "person", "bicycle", "motorcycle", "car",           # People & mobility
    "bench", "chair", "table", "backpack", "laptop",    # Furniture
    "cell phone",                                        # Tech
    "door", "openedDoor", "window", "stairs",           # Navigation
    "step", "ramp", "pole", "corridor",                 # Hazards
]
```

**High-priority classes** (immediate TTS interrupt, bypass queue):
```python
HIGH_PRIORITY_OBJECTS = {"stairs", "step", "ramp", "openedDoor", "pole"}
```

**Danger zone** (tightened in March 2025 overhaul):
- X: 28% of frame width (was 35%)
- Y: 40% of frame height (was 50%)
- Reduces false positives from periphery objects

### mBLIP — Multilingual BLIP-2 for Native Telugu

**Model:** `Gregor/mblip-mt0-xl` (multilingual BLIP-2, 96 languages including Telugu)
**Adapter:** LoRA on language model layers only (`["q", "v"]` attention matrices)
**Prompt:** `"క్లుప్తంగా వివరించు:"` (Describe briefly in Telugu)

**Why mBLIP over standard BLIP:**
- Standard BLIP needs fine-tuning on a Telugu dataset → all free Telugu datasets deleted/private by March 2026
- mBLIP generates Telugu **zero-shot** — no Telugu dataset required for basic operation
- LoRA adds campus-specific vocabulary with only 300–500 photos

### Priority TTS System

Three levels (implemented in `audio_module.py`):
1. **HIGH priority** (`tts.alert()`) — Danger alerts (stairs/poles) — clears queue, interrupts immediately
2. **LOW priority** (`tts.speak()`) — Scene descriptions — plays in order
3. OCR results → HIGH priority (sign text always interrupts)

**Backends:**
- `edge-tts` (default) — Microsoft Neural Telugu voice `te-IN-ShrutiNeural` (requires internet)
- `pyttsx3` (fallback) — Offline English only (no Telugu neural voice offline)

### Spatial Analysis System

`SpatialAnalyzer` in `vision_module.py` converts bounding box coords to:
- **Clock position** (3×3 grid: 12 o'clock = straight ahead)
- **Distance word**: very close (bbox area >15% of frame), nearby (3–15%), in the distance (<3%)
- **Danger zone flag** (centre 28% × 40% of frame)

### OCR Auto-Trigger

OCR fires automatically (no user action) when YOLO detects:
```python
OCR_AUTO_TRIGGER_CLASSES = {"sign", "notice", "board", "poster", "text"}
```
With minimum confidence `OCR_AUTO_TRIGGER_CONFIDENCE = 0.60`.

Auto-trigger has a `_AUTO_COOLDOWN = 5.0` second cooldown to avoid spam. Manual toggle via `R` key.

---

## March 2025 Overhaul — Bugs Fixed & Logic Improved

### Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `training/train_detector.py` | `DATA_DIR_REF` used before assignment (silent crash) | Removed; use `DATA_DIR` directly from config import |
| `training/train_captioner.py` | `from torch.cuda.amp import GradScaler, autocast` (deprecated PyTorch API) | Replaced with `torch.amp.GradScaler(device=...)` and `torch.amp.autocast(device_type=...)` |
| `training/evaluate.py` | Used `VIZWIZ_DIR` (removed from config) | Updated to use `CAMPUS_CAPTION_DIR` |

### Logic Improvements

| Feature | Before | After |
|---------|--------|-------|
| YOLO classes | 80 COCO classes | 18 campus-specific classes |
| Danger zone | 35% × 50% of frame | 28% × 40% (tighter, fewer false positives) |
| OCR trigger | Manual 'R' key only | Auto-triggers on sign/board/notice YOLO detection |
| High-priority objects | None | `stairs`, `step`, `ramp`, `openedDoor`, `pole` → immediate `tts.alert()` |
| Detection sort order | Danger zone first, then confidence | High-priority first, then danger zone, then confidence |
| Bbox colouring | Green / Red | Green / Orange (high-priority) / Red (danger zone) |

---

## March 2026 — mBLIP Migration

### Problem

`Hardik15/telugu-image-captions` (added during March 2025 overhaul) confirmed **deleted / 404** on HuggingFace. All alternative Telugu image-caption datasets verified unavailable:

| Dataset | Status |
|---------|--------|
| `Hardik15/telugu-image-captions` | ❌ 404 Deleted |
| `FutureBeeAI/telugu-image-captions` | ❌ 404 Deleted |
| `Telugu-LLM-Labs/Telugu-Image-Captions` | ❌ 404 Private |
| `ai4bharat/IndicCOCO` | ❌ 404 Deleted |
| `Telugu-LLM-Labs/Indic-Multimodal-Instructions` | ❌ 404 Not accessible |
| `dinhanhx/crossmodal-3600` | ✅ Live but only 3,600 images — too small |
| `visheratin/laion-coco-nllb` | ✅ Live but machine-translated Telugu only |

**Conclusion:** No free, publicly available, native Telugu image caption dataset exists.

### Solution: Switch to mBLIP

Migrated from `Salesforce/blip-image-captioning-base` to `Gregor/mblip-mt0-xl`:
- mBLIP trained on 96 languages including Telugu natively
- Telugu output zero-shot (no Telugu training data needed for basic operation)
- LoRA fine-tuning on 300–500 human-collected campus photos sufficient for campus-specific vocabulary

### Files Changed in mBLIP Migration

| File | Change |
|------|--------|
| `config.py` | Complete rewrite — mBLIP settings, LoRA params, 4-bit flag, MBLIP_PROMPT |
| `data/download_datasets.py` | Replaced HF Telugu download with campus dataset folder setup utility |
| `data/dataset_loader.py` | Replaced `TeluguCaptionDataset` with `CampusCaptionDataset` (Blip2Processor) |
| `training/train_captioner.py` | Complete rewrite — mBLIP + LoRA fine-tuning (saves adapter only, ~100 MB) |
| `training/evaluate.py` | Updated for Blip2Processor, MBLIP_PROMPT, LoRA adapter auto-detection |
| `src/caption_module.py` | Swapped BlipProcessor/BlipForConditionalGeneration → Blip2 equivalents + LoRA loading |
| `requirements.txt` | Added `peft>=0.10.0`, `bitsandbytes>=0.43.0` |
| `README.md` | Full rewrite — mBLIP architecture table, updated dataset table + project structure |
| `TRAINING_GUIDE.md` | Full rewrite — mBLIP LoRA training guide, cloud training tips |
| `NEXT_STEPS_DATASET_UPGRADE.md` | Updated to mBLIP action plan |
| `DATASET_CREATION_GUIDE.md` | **NEW** — Complete guide for creating human campus caption dataset |
| `data/campus_captions/train.json` | **NEW** — Sample JSON with 2 placeholder entries |
| `data/campus_captions/val.json` | **NEW** — Sample JSON with 1 placeholder entry |

### Files NOT Changed in mBLIP Migration

- `main.py` — No changes needed; mBLIP used inside `SceneCaptioner` only
- `src/vision_module.py` — YOLO pipeline unchanged
- `src/audio_module.py` — TTS unchanged
- `src/ocr_module.py` — OCR unchanged
- `training/train_detector.py` — YOLO training unchanged
- `data/augmentations.py` — Image transforms work with any model
- `demo/` — Untouched (separate standalone system using Gemini API)
- `tests/` — Tests relate to audio/spatial modules; unchanged

---

## FastAPI Mobile Architecture (api.py)

The FastAPI server provides a mobile deployment path:

```
Mobile Phone Camera
      │ (base64 JPEG @ 8 FPS)
      ▼
POST /analyze   (FastAPI endpoint)
      │
      ├─► YOLO11 inference (synchronous, ~30ms)
      │     └─► SpatialAnalyzer → clock_pos, distance_word, in_danger_zone, is_high_priority
      │
      ├─► SceneCaptioner.caption() (BackgroundTask — async, runs if 4s elapsed)
      │     └─► mBLIP generates Telugu caption
      │
      └─► Response (JSON):
            { "caption": "<Telugu scene description>",
              "alerts": ["<priority alert strings>"],
              "detections": [...] }

      ▼
Frontend (frontend/index.html)
      │ window.speechSynthesis.speak(utterance, lang='te-IN')
      │ Alerts interrupt (cancel() + immediate speak)
      └─► User hears Telugu descriptions on phone speaker
```

**Key design choices in api.py:**
- YOLO runs synchronously (fast, ~30ms) — always returns latest detections
- mBLIP runs in `BackgroundTasks` — never blocks the HTTP response
- Response always includes `captioner.last_caption` (most recent, may be a few seconds old)
- Alerts generated for all `is_high_priority` or `in_danger_zone` detections

---

## Inference Speed Optimizations (caption_module.py)

Five techniques applied to fit mBLIP on 4 GB VRAM and keep latency low:

1. **4-bit NF4 quantization** — `bitsandbytes` reduces VRAM from ~8 GB to ~3.5 GB
2. **Greedy search** — `num_beams=1, do_sample=False` (fastest generation)
3. **Max 25 tokens** — Navigation captions don't need to be long
4. **224×224 image resize** — Before Q-Former encoding, reduces compute
5. **Scene change detection** — `cv2.absdiff()` on grayscale frames; if `mean_diff < 15.0`, reuse last caption

---

## Known Issues & Notes

### export_models.py Uses Old BLIP Names

`training/export_models.py` still imports `BLIP_FINETUNED_PATH` and `BLIP_ONNX_PATH` from `config.py` — these are aliases for the mBLIP paths. The export of mBLIP vision encoder to ONNX works but the full language model (mT0-xl) cannot be directly exported to ONNX due to the generative architecture. Only the vision encoder export is functional.

**Impact:** ONNX/OpenVINO export of the full mBLIP pipeline is not yet complete. YOLO export to ONNX works fine.

### MANUAL_DOWNLOADS.md References Old Telegu Dataset

`data/MANUAL_DOWNLOADS.md` still shows a folder reference to `data/telugu_captions/` (from the March 2025 overhaul). This folder is now empty — the correct dataset path is `data/campus_captions/`. The folder summary in MANUAL_DOWNLOADS.md needs updating.

### Demo README References Old BLIP Flags

`demo/README_DEMO.md` Step "After Testing" references:
```python
BLIP_USE_FINETUNED = True   # ← old flag name
```
The correct flag in `config.py` is `MBLIP_USE_FINETUNED = True`.

---

## Verification Checklist (Before Training)

- [ ] Python 3.10+ installed
- [ ] PyTorch with CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `pip install -r requirements.txt` completed without critical errors
- [ ] `data/campus_captions/train.json` has >2 entries (your photos, not sample data)
- [ ] `data/campus_captions/images/` contains corresponding `.jpg` files
- [ ] At least one Roboflow dataset extracted in `data/indoor_campus/`
- [ ] `python data/download_datasets.py --dataset verify` shows ✓ for both
- [ ] Internet connection available (for mBLIP HuggingFace download ~5 GB + edge-tts)

---

## Verification Checklist (After Training)

- [ ] `checkpoints/yolo11_campus.pt` exists (YOLO training complete)
- [ ] `checkpoints/mblip_campus/best/adapter_config.json` exists (LoRA training complete)
- [ ] `python training/evaluate.py` shows BLEU-4 ≥ 0.25 and METEOR ≥ 0.28
- [ ] `config.py` updated: `YOLO_USE_CUSTOM = True`, `MBLIP_USE_FINETUNED = True`
- [ ] `python main.py` starts and announces "Campus navigation system ready."
