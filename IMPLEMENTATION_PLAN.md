# Implementation Plan Archive — March 2025 Overhaul

This document records all changes made during the comprehensive March 2025 overhaul of Blind-Project.

---

## Why the Overhaul Was Needed

Three critical problems were identified:

1. **Unavailable Datasets** — `ai4bharat/IndicCOCO` was confirmed unavailable on HuggingFace. The fallback datasets (MS-COCO 18 GB, VizWiz) were too large and contained irrelevant objects.
2. **Code Bugs** — `DATA_DIR_REF` variable in `train_detector.py` was used before it was defined (silent crash). Deprecated `torch.cuda.amp.autocast` in `train_captioner.py`.
3. **Suboptimal Logic** — All 80 COCO classes were trained (elephants, pizzas, etc.). OCR only fired on manual 'R' press. Danger zone covered too much of the frame.

---

## Changes Made

### Datasets Replaced

| Old | New |
|-----|-----|
| `ai4bharat/IndicCOCO` (unavailable) | `Hardik15/telugu-image-captions` (HuggingFace, ~25K pairs, Aug 2024) |
| MS-COCO 2017 (18 GB, 80 classes) | Roboflow campus-specific datasets |
| VizWiz-Captions (noisy, English) | Removed entirely |

### Bugs Fixed

| File | Bug | Fix |
|------|-----|-----|
| `training/train_detector.py` | `DATA_DIR_REF` used before assignment | Removed; use `DATA_DIR` directly from config import |
| `training/train_captioner.py` | `from torch.cuda.amp import GradScaler, autocast` (deprecated) | Replaced with `torch.amp.GradScaler(device=...)` and `torch.amp.autocast(device_type=...)` |
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

### Files Changed

| File | Change Type |
|------|-------------|
| `config.py` | Rewrite — new dataset paths, campus classes, danger zone, priority objects |
| `data/download_datasets.py` | Rewrite — Telugu HF dataset, removed VizWiz/COCO/IndicCOCO |
| `data/dataset_loader.py` | Rewrite — `TeluguCaptionDataset` + `CampusDetectionVerifier` |
| `data/MANUAL_DOWNLOADS.md` | Rewrite — exact Roboflow URLs |
| `training/train_detector.py` | Rewrite — bug fix, campus-only 18 classes |
| `training/train_captioner.py` | Rewrite — bug fix, Telugu-only dataset |
| `training/evaluate.py` | Rewrite — Telugu eval, fixed deprecated imports |
| `src/vision_module.py` | Update — high-priority flag, orange bbox, updated docstrings |
| `src/caption_module.py` | Update — expanded PRIORITY_OBJECTS, high-priority alerts |
| `src/audio_module.py` | Update — latency note, post-fine-tune note |
| `src/ocr_module.py` | Update — OCR auto-trigger from YOLO detections |
| `main.py` | Rewrite — high-priority alert path, OCR auto-trigger integration |
| `requirements.txt` | Update — added huggingface-hub, deep-translator, pygame; removed pycocotools |
| `README.md` | Rewrite — new pipeline documentation |
| `TRAINING_GUIDE.md` | Rewrite — 6-phase training guide |
| `NEXT_STEPS_DATASET_UPGRADE.md` | Rewrite — post-training steps |

### Files NOT Changed (Intentionally)
- `demo/` — All demo files untouched (separate system)
- `tests/` — All unit tests pass without changes
- `data/augmentations.py` — Compatible with new dataset loader
- `training/export_models.py` — Still valid for ONNX export
