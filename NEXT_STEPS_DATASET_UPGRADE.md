# Next Steps After Training — Blind-Project

> **Updated March 2025.** After the dataset overhaul is complete and training is done, here are the recommended next steps.

---

## Immediate Post-Training Steps

### 1. Enable Custom Models in config.py
```python
YOLO_USE_CUSTOM    = True    # Load checkpoints/yolo11_campus.pt
BLIP_USE_FINETUNED = True    # Load checkpoints/blip_telugu/best/
```

### 2. Test on Live Camera
```bash
python main.py
```
Walk through:
- A **doorway** → confirm Telugu alert fires immediately
- Near **stairs** → confirm high-priority interrupt
- Hold a **sign** → confirm OCR auto-triggers without pressing 'R'

### 3. Evaluate Metrics
```bash
python training/evaluate.py --max-samples 500
```
Check `logs/eval_report.json` — target: BLEU-4 ≥ 0.25, METEOR ≥ 0.28.

---

## If Metrics Are Low

| Problem | Fix |
|---------|-----|
| BLEU-4 < 0.20 | Train more epochs: `--epochs 12` |
| YOLO misses stairs | Add more stair images: re-download larger Roboflow dataset |
| Poor Telugu pronunciation | Check edge-tts voice: `EDGE_TTS_VOICE = "te-IN-MohanNeural"` (male alt) |

---

## Optional Upgrades

### A. Upgrade to YOLO11m (Higher Accuracy)
If your GPU has 8+ GB VRAM:
```python
# In config.py:
YOLO_MODEL_NAME = "yolo11m.pt"   # ~8% higher mAP than yolo11s
```
Retrain: `python training/train_detector.py --model yolo11m.pt`

### B. Offline TTS (No Internet Required)
For fully offline operation:
```python
# In config.py:
TTS_ENGINE = "pyttsx3"     # English only, but no internet needed
TELUGU_MODE = False         # Revert to English for offline
```

### C. Edge Deployment (Export to ONNX)
```bash
python training/export_models.py
```
Loads BLIP from `checkpoints/blip_telugu/best/` and exports ONNX for faster CPU inference.

---

## What Was Fixed in the March 2025 Overhaul

| Issue | Status |
|-------|--------|
| `ai4bharat/IndicCOCO` unavailable | ✅ Replaced with `Hardik15/telugu-image-captions` |
| Full MS-COCO (18 GB, 80 classes) | ✅ Removed — campus-only 18 classes now |
| VizWiz noisy blind-user photos | ✅ Removed |
| `DATA_DIR_REF` crash bug in train_detector.py | ✅ Fixed |
| Deprecated `torch.cuda.amp.autocast` | ✅ Fixed → `torch.amp.autocast` |
| OCR required manual 'R' key press | ✅ Auto-triggers on sign/board/notice detection |
| Danger zone too wide (35%×50%) | ✅ Tightened to 28%×40% |
| Stairs/doors not prioritised | ✅ HIGH_PRIORITY_OBJECTS → immediate TTS interrupt |
