"""
config.py — Central configuration for Blind-Project.
All constants, paths, thresholds, and runtime settings live here.
Edit this file before training or running the application.

═══════════════════════════════════════════════════════════════
  PROJECT GOAL
═══════════════════════════════════════════════════════════════
  Build an assistive navigation app for blind users on a
  college / university campus that:
    Phase 1 — Detects 18 campus-relevant objects in real time
               (YOLO11s fine-tuned on Roboflow indoor datasets)
    Phase 2 — Describes the scene natively in TELUGU language
               (mBLIP — multilingual BLIP-2 supporting 96 languages
                including Telugu natively; fine-tuned on human
                campus caption dataset)
    Output  — Telugu voice via Microsoft Neural TTS (edge-tts)
               or offline pyttsx3 fallback

  Dataset strategy (March 2026):
    YOLO  → Roboflow campus-specific datasets (door/stairs/pole …)
    mBLIP → Human-collected campus caption dataset (your own photos
             + Telugu captions — see DATASET_CREATION_GUIDE.md)
    NO HuggingFace Telugu datasets — all confirmed deleted/unavailable
═══════════════════════════════════════════════════════════════
"""

import os
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.resolve()
DATA_DIR        = ROOT_DIR / "data"

# Caption dataset for mBLIP fine-tuning (human campus photos + Telugu captions)
# See DATASET_CREATION_GUIDE.md for how to create this dataset
CAMPUS_CAPTION_DIR  = DATA_DIR / "campus_captions"   # Your own campus images + Telugu captions

# Object detection dataset for YOLO training (campus-specific, Roboflow)
INDOOR_DIR          = DATA_DIR / "indoor_campus"      # Manual download — see data/MANUAL_DOWNLOADS.md

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR        = ROOT_DIR / "logs"
MODELS_DIR      = ROOT_DIR / "models"
EXPORTED_DIR    = ROOT_DIR / "exported_models"
TESTS_DIR       = ROOT_DIR / "tests"

# Create directories if they don't exist
for _dir in [DATA_DIR, CAMPUS_CAPTION_DIR, INDOOR_DIR,
             CHECKPOINTS_DIR, LOGS_DIR, MODELS_DIR, EXPORTED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16    = (DEVICE == "cuda")          # Mixed precision on GPU
NUM_WORKERS = 4 if DEVICE == "cuda" else 0

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0          # Change if using external camera
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30
FRAME_BUFFER_SIZE = 5          # Max frames held in buffer

# ─────────────────────────────────────────────────────────────────────────────
# API CONFIGURATION (Mobile Delivery)
# ─────────────────────────────────────────────────────────────────────────────
API_HOST          = "0.0.0.0"
API_PORT          = 8000
API_CORS_ORIGINS  = ["*"]      # Allow all origins for the frontend app


# ─────────────────────────────────────────────────────────────────────────────
# YOLO OBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# yolo11s — small model, good accuracy, suitable for 4–6 GB VRAM
# Upgrade tip: switch to yolo11m.pt if you have 8+ GB VRAM (~8% higher mAP)
YOLO_MODEL_NAME       = "yolo11s.pt"
YOLO_CUSTOM_WEIGHTS   = str(CHECKPOINTS_DIR / "yolo11_campus.pt")
YOLO_CONFIDENCE       = 0.40
YOLO_IOU_THRESHOLD    = 0.45
YOLO_IMG_SIZE         = 640
YOLO_USE_CUSTOM       = False    # Set True after campus fine-tuning

# ─────────────────────────────────────────────────────────────────────────────
# CAMPUS-SPECIFIC OBJECT CLASSES
# ─────────────────────────────────────────────────────────────────────────────
# ONLY classes relevant to a college/university campus environment.
# 18 classes — avoids "class confusion" from irrelevant COCO objects.
CAMPUS_CLASSES = [
    # People & mobility
    "person", "bicycle", "motorcycle",
    # Vehicles (relevant near campus roads)
    "car",
    # Campus furniture & navigation
    "bench", "chair", "table", "backpack", "laptop",
    # Building navigation (CRITICAL for blind users)
    "door", "openedDoor", "window", "stairs", "step", "ramp",
    # Hazards
    "pole", "corridor",
    # Tech
    "cell phone",
]

# Objects that ALWAYS trigger a HIGH-PRIORITY audio alert (interrupt current speech)
HIGH_PRIORITY_OBJECTS = {
    "stairs", "step", "ramp", "openedDoor", "pole",
}

# Objects that auto-trigger OCR when detected with high confidence
OCR_AUTO_TRIGGER_CLASSES = {
    "sign", "notice", "board", "poster", "text",
}
OCR_AUTO_TRIGGER_CONFIDENCE = 0.60   # Min confidence to auto-trigger OCR

# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL REASONING
# ─────────────────────────────────────────────────────────────────────────────
# Tightened danger zone for precise obstacle detection (was 35%×50%)
DANGER_ZONE_X_RATIO       = 0.28   # 28% of frame width (tighter centre band)
DANGER_ZONE_Y_RATIO       = 0.40   # 40% of frame height (tighter centre band)
DISTANCE_NEAR_THRESHOLD   = 0.15   # bbox area > 15% → "very close"
DISTANCE_MEDIUM_THRESHOLD = 0.03   # bbox area 3–15% → "nearby"

# ─────────────────────────────────────────────────────────────────────────────
# mBLIP CAPTIONING MODEL
# ─────────────────────────────────────────────────────────────────────────────
# mBLIP = multilingual BLIP-2 supporting 96 languages including Telugu natively.
# Model: Gregor/mblip-mt0-xl on HuggingFace (~5 GB download on first run)
# Paper: https://aclanthology.org/2023.emnlp-main.648/
#
# VRAM requirements:
#   MBLIP_USE_4BIT = True  → ~3.5 GB VRAM  (RTX 3050 4GB) ← DEFAULT for this project
#   MBLIP_USE_4BIT = False → ~8–10 GB VRAM (Colab T4 / Kaggle P100 / RTX 3080+)
#
# Fine-tuning uses LoRA — only trains a small adapter (~100 MB).
# The base mBLIP weights are FROZEN (not changed during training).
MBLIP_PRETRAINED_NAME  = "Gregor/mblip-mt0-xl"
MBLIP_FINETUNED_PATH   = str(CHECKPOINTS_DIR / "mblip_campus")
MBLIP_USE_FINETUNED    = False     # Set True after campus fine-tuning completes

# ── 4-bit Quantization (for 4 GB VRAM GPUs like RTX 3050) ───────────────────
# True  = uses bitsandbytes int8/4-bit → fits 4 GB VRAM, tiny quality loss
# False = uses float16 → better quality, needs 8–12 GB VRAM (use on Colab/cloud)
MBLIP_USE_4BIT         = True      # Recommended: True for RTX 3050 4GB

# ── Caption Generation ───────────────────────────────────────────────────────
# Prompt sent to mBLIP. The model uses this to know to respond in Telugu.
MBLIP_PROMPT           = "క్లుప్తంగా వివరించు:"
MBLIP_LANGUAGE         = "Telugu"  # Informational / for logging
MBLIP_MAX_NEW_TOKENS   = 25        # Navigation captions don't need to be long
MBLIP_NUM_BEAMS        = 1         # Greedy search for fastest generation
MBLIP_CAPTION_INTERVAL = 4.0       # Seconds between full scene captions
SCENE_CHANGE_THRESHOLD = 15.0      # Skip mBLIP if avg pixel diff is below this

# ── LoRA Fine-tuning Hyperparams ─────────────────────────────────────────────
# LoRA only updates a small number of parameters — fits 4 GB VRAM for training.
MBLIP_LORA_RANK        = 16        # LoRA rank (higher = more params, better quality)
MBLIP_LORA_ALPHA       = 32        # LoRA scaling factor (usually 2× rank)
MBLIP_LORA_DROPOUT     = 0.05      # LoRA dropout

# ── Backward-compatibility alias (used in a few old references) ──────────────
BLIP_CAPTION_INTERVAL  = MBLIP_CAPTION_INTERVAL

# ONNX / OpenVINO paths (populated after export — optional)
MBLIP_ONNX_PATH        = str(EXPORTED_DIR / "mblip_vision_encoder.onnx")
OPENVINO_MODEL_DIR     = str(EXPORTED_DIR / "mblip_openvino")

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES  = ["en", "te"]     # Telugu + English OCR
OCR_GPU        = (DEVICE == "cuda")
OCR_CONFIDENCE = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH  —  Telugu Neural Voice
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: edge-tts requires an active internet connection.
#       Latency: 800–1500ms per utterance (network dependent).
#       For offline use, pyttsx3 backend is used (English only).
#       Set TTS_ENGINE = "pyttsx3" for fully offline English operation.
TTS_ENGINE        = "edge-tts"            # "edge-tts" or "pyttsx3"
TTS_RATE          = 175
TTS_VOLUME        = 1.0
TTS_VOICE_GENDER  = "female"
EDGE_TTS_VOICE    = "te-IN-ShrutiNeural"  # Telugu female neural voice (Microsoft)

# Telugu output flag — mBLIP outputs Telugu natively (no translation needed)
# Set False only if you want to revert to English output (testing only)
TELUGU_MODE       = True

# Priority levels for TTS queue
TTS_PRIORITY_HIGH = 0       # Danger alerts — interrupt immediately
TTS_PRIORITY_LOW  = 1       # Scene descriptions — plays in order

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — mBLIP CAPTIONER (LoRA fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Training only fine-tunes the LoRA adapter (not the full mBLIP model).
#       Only 1–3 epochs are usually needed since mBLIP already knows Telugu.
#
# For RTX 3050 4 GB:
#   MBLIP_TRAIN_BATCH_SIZE = 1
#   MBLIP_GRAD_ACCUM_STEPS = 8  → effective batch = 8
#
# For Colab T4 / Kaggle (15–16 GB VRAM):
#   MBLIP_TRAIN_BATCH_SIZE = 4
#   MBLIP_GRAD_ACCUM_STEPS = 4  → effective batch = 16
MBLIP_TRAIN_EPOCHS        = 3          # 3 epochs is usually enough for LoRA
MBLIP_TRAIN_BATCH_SIZE    = 1          # RTX 3050 4 GB: must be 1
MBLIP_LEARNING_RATE       = 2e-4       # LoRA standard learning rate
MBLIP_WEIGHT_DECAY        = 0.01
MBLIP_WARMUP_STEPS        = 50
MBLIP_GRAD_ACCUM_STEPS    = 8          # Effective batch = 1 × 8 = 8
MBLIP_SAVE_STEPS          = 50
MBLIP_EVAL_STEPS          = 50

# Training sample limits
MBLIP_MAX_TRAIN_SAMPLES   = None       # Use all campus images (your dataset is small)
MBLIP_MAX_VAL_SAMPLES     = None

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — YOLO DETECTOR  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────
YOLO_TRAIN_EPOCHS        = 80
YOLO_TRAIN_BATCH_SIZE    = 16
YOLO_TRAIN_IMG_SIZE      = 640
YOLO_TRAIN_LR0           = 0.01
YOLO_TRAIN_PATIENCE      = 15        # Early stopping patience

# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPER WINDOW
# ─────────────────────────────────────────────────────────────────────────────
DEV_WINDOW_TITLE         = "Blind-Project | Campus Navigation"
SHOW_DEV_WINDOW          = True
BBOX_THICKNESS           = 2
FONT_SCALE               = 0.55
COLOR_BBOX_DEFAULT       = (0, 200, 0)
COLOR_BBOX_DANGER        = (0, 0, 255)
COLOR_BBOX_HIGH_PRIORITY = (0, 128, 255)  # Orange — for stairs/poles
COLOR_DANGER_ZONE        = (0, 0, 200)
COLOR_TEXT_BG            = (0, 0, 0)
COLOR_TEXT_FG            = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = str(LOGS_DIR / "blind_project.log")
