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

---

## Telugu Caption Dataset Investigation — March 2026

### Problem

`Hardik15/telugu-image-captions` (the dataset added during the March 2025 overhaul) was confirmed **deleted / 404** on HuggingFace as of March 2026. This means `python data/download_datasets.py --dataset telugu` will fail.

### All Candidates Checked (Live Browser Verification)

| Dataset | Source | Native Telugu? | Status | Verdict |
|---------|--------|---------------|--------|---------|
| `Hardik15/telugu-image-captions` | HuggingFace | Unknown | ❌ **404 Deleted** | Original dataset — gone |
| `FutureBeeAI/telugu-image-captions` | HuggingFace | Yes | ❌ **404 Deleted** | Removed |
| `Telugu-LLM-Labs/Telugu-Image-Captions` | HuggingFace | Unknown | ❌ **404 Private** | Not accessible |
| `ai4bharat/IndicCOCO` | HuggingFace | Yes | ❌ **404 Deleted** | Confirmed gone again |
| `Telugu-LLM-Labs/Indic-Multimodal-Instructions` | HuggingFace | Mixed | ❌ **404** | Not accessible |
| `gksriharsha/chitralekha` | HuggingFace | No | ❌ **Wrong type** | Telugu OCR dataset — not scene captions |
| `ai4bharat/sangraha` | HuggingFace | No | ❌ **No images** | Text-only LLM dataset |
| `visheratin/laion-coco-nllb` | HuggingFace | No (NLLB translated) | ✅ Live | 893K pairs but machine-translated, not native |
| `wikimedia/wit_base` | HuggingFace | Mixed | ✅ Live | Wikipedia images, limited Telugu captions |
| `dinhanhx/crossmodal-3600` | HuggingFace | Yes | ✅ Live | Only 3,600 images — too small for fine-tuning |
| `FutureBeeAI` (website) | futurebeeai.com | **Yes — Native** | ✅ Live | **Paid / commercial** — 25,000+ pairs, 100+ native speakers |

### Key Finding

> **There is currently NO free, publicly available, native Telugu image caption dataset on HuggingFace, GitHub, or Kaggle.**

All previously referenced datasets have been deleted or are private. Machine-translated Telugu datasets exist but do not produce natural-sounding output for a blind navigation assistant.

### Recommended Path Forward (3 Options)

#### Option A — FutureBeeAI Commercial Dataset *(Best quality, paid)*
- **URL:** https://www.futurebeeai.com/dataset/multi-modal-dataset/telugu-image-caption-dataset
- 25,000+ images with native Telugu captions written by 100+ native speakers
- Directly suitable for BLIP fine-tuning
- Requires contacting them for pricing / access
- **Best choice if budget is available**

#### Option B — Collect Your Own Campus Telugu Captions *(Best for this project, free)*
- Take 500–1,000 photos on your actual college campus (doors, stairs, corridors, people, chairs, signs)
- Have native Telugu speakers write one or two caption sentences per image
- Use Label Studio (free tool) to organise: https://labelstud.io
- Even 500 real campus images with Telugu descriptions will teach BLIP exactly the right vocabulary and environment
- **Best choice for campus-specific accuracy** — no existing dataset covers this scenario
- Output: place results in `data/telugu_captions/` in the existing `train.json` / `val.json` format

#### Option C — AI4Bharat Website *(Check directly)*
- AI4Bharat may host IndicCOCO or a replacement outside HuggingFace
- Check directly: https://ai4bharat.iitm.ac.in/
- If available, it would be the closest academic-quality free option

### Resolution

**Migrated from standard BLIP to mBLIP (March 2026)** — see section below.

---

## mBLIP Migration — March 2026

### Why mBLIP?

All Telugu image-caption datasets on HuggingFace confirmed deleted/unavailable (see previous section).
Solution: switched to `Gregor/mblip-mt0-xl` — a BLIP-2 model trained on 96 languages **including Telugu natively**.
mBLIP can describe campus scenes in Telugu **zero-shot** — no Telugu dataset required.
Only a small human-collected campus dataset (300–500 photos) is needed for LoRA domain adaptation.

### Hardware Configuration

Project configured for: **RTX 3050 4GB VRAM**
- `MBLIP_USE_4BIT = True` (4-bit NF4 quantization via bitsandbytes)
- `MBLIP_TRAIN_BATCH_SIZE = 1`, `MBLIP_GRAD_ACCUM_STEPS = 8`
- For cloud (Colab/Kaggle): use `--no-4bit --batch-size 4`

### Files Changed

| File | Change |
|------|--------|
| `config.py` | Complete rewrite — mBLIP settings, LoRA params, 4-bit flag, MBLIP_PROMPT |
| `data/download_datasets.py` | Replaced HF Telugu download with campus dataset folder setup utility |
| `data/dataset_loader.py` | Replaced `TeluguCaptionDataset` with `CampusCaptionDataset` (Blip2Processor) |
| `training/train_captioner.py` | Complete rewrite — mBLIP + LoRA fine-tuning (saves adapter only, ~100 MB) |
| `training/evaluate.py` | Updated for Blip2Processor, MBLIP_PROMPT, LoRA adapter auto-detection |
| `src/caption_module.py` | Swapped BlipProcessor/BlipForConditionalGeneration for Blip2 equivalents + LoRA loading |
| `requirements.txt` | Added `peft>=0.10.0`, `bitsandbytes>=0.43.0` |
| `README.md` | Full rewrite — mBLIP architecture table, updated dataset table + project structure |
| `TRAINING_GUIDE.md` | Full rewrite — mBLIP LoRA training guide, cloud training tips |
| `NEXT_STEPS_DATASET_UPGRADE.md` | Updated to mBLIP action plan |
| `DATASET_CREATION_GUIDE.md` | **NEW** — Complete guide for creating human campus caption dataset |
| `data/campus_captions/train.json` | **NEW** — Sample JSON (replace with your data) |
| `data/campus_captions/val.json` | **NEW** — Sample JSON (replace with your data) |

### Files NOT Changed

- `main.py` — No changes needed; mBLIP used inside `SceneCaptioner` only
- `src/vision_module.py` — YOLO pipeline unchanged
- `src/audio_module.py` — TTS unchanged
- `src/ocr_module.py` — OCR unchanged
- `training/train_detector.py` — YOLO training unchanged
- `data/augmentations.py` — Image transforms work with any model
- `demo/` — Untouched (separate standalone system)
- `tests/` — Tests relate to audio/spatial modules; unchanged
