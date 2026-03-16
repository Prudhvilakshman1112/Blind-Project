"""
config.py — Central configuration for Blind-Project.
All constants, paths, thresholds, and runtime settings live here.
Edit this file before training or running the application.

═══════════════════════════════════════════════════════════════
  PROJECT GOAL
═══════════════════════════════════════════════════════════════
  Build an assistive navigation app for blind users that:
    Phase 1 — Detects objects in university/indoor premises
              (YOLO trained on COCO + indoor campus datasets)
    Phase 2 — Describes the scene natively in TELUGU language
              (BLIP fine-tuned on AI4Bharat IndicCaption Telugu)
    Output  — Telugu voice via Microsoft Neural TTS
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

# Caption datasets (for BLIP training)
VIZWIZ_DIR      = DATA_DIR / "vizwiz"           # English blind-user captions
COCO_DIR        = DATA_DIR / "coco"             # English COCO captions
INDIC_DIR       = DATA_DIR / "indic_caption"    # Telugu captions (AI4Bharat)

# Object detection datasets (for YOLO training)
INDOOR_DIR      = DATA_DIR / "indoor_campus"    # Manual download — see data/MANUAL_DOWNLOADS.md

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR        = ROOT_DIR / "logs"
MODELS_DIR      = ROOT_DIR / "models"
EXPORTED_DIR    = ROOT_DIR / "exported_models"
TESTS_DIR       = ROOT_DIR / "tests"

# Create directories if they don't exist
for _dir in [DATA_DIR, VIZWIZ_DIR, COCO_DIR, INDIC_DIR, INDOOR_DIR,
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
# YOLO OBJECT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# yolo11s (small) is used for training — significantly more accurate than nano.
# yolo11n (nano) was only used in the demo for speed.
YOLO_MODEL_NAME       = "yolo11s.pt"             # small — better accuracy for final app
YOLO_CUSTOM_WEIGHTS   = str(CHECKPOINTS_DIR / "yolo11_custom.pt")
YOLO_CONFIDENCE       = 0.40                     # Lowered slightly — model is more accurate
YOLO_IOU_THRESHOLD    = 0.45
YOLO_IMG_SIZE         = 640
YOLO_USE_CUSTOM       = False                    # Set True after fine-tuning

# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL REASONING
# ─────────────────────────────────────────────────────────────────────────────
DANGER_ZONE_X_RATIO       = 0.35   # 35% of frame width centered
DANGER_ZONE_Y_RATIO       = 0.50   # 50% of frame height centered
DISTANCE_NEAR_THRESHOLD   = 0.15   # bbox area > 15% → "very close"
DISTANCE_MEDIUM_THRESHOLD = 0.05   # bbox area 5–15% → "nearby"

# ─────────────────────────────────────────────────────────────────────────────
# BLIP CAPTIONING MODEL
# ─────────────────────────────────────────────────────────────────────────────
BLIP_PRETRAINED_NAME   = "Salesforce/blip-image-captioning-base"
BLIP_FINETUNED_PATH    = str(CHECKPOINTS_DIR / "blip_finetuned")
BLIP_USE_FINETUNED     = False     # Set True after fine-tuning

BLIP_MAX_NEW_TOKENS    = 80        # Slightly more for Telugu sentences
BLIP_NUM_BEAMS         = 4
BLIP_CAPTION_INTERVAL  = 4.0      # Seconds between full scene captions

# ONNX / OpenVINO paths (populated after export)
BLIP_ONNX_PATH        = str(EXPORTED_DIR / "blip_vision_encoder.onnx")
OPENVINO_MODEL_DIR    = str(EXPORTED_DIR / "blip_openvino")

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES  = ["en", "te"]     # Telugu + English OCR
OCR_GPU        = (DEVICE == "cuda")
OCR_CONFIDENCE = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH  —  Telugu Neural Voice
# ─────────────────────────────────────────────────────────────────────────────
TTS_ENGINE        = "edge-tts"            # edge-tts supports Telugu neural voice
TTS_RATE          = 175
TTS_VOLUME        = 1.0
TTS_VOICE_GENDER  = "female"
EDGE_TTS_VOICE    = "te-IN-ShrutiNeural" # Telugu female neural voice (Microsoft)

# Telugu output flag — all spoken text translated to Telugu before TTS
TELUGU_MODE       = True    # Set False to revert to English output

# Priority levels for TTS queue
TTS_PRIORITY_HIGH = 0       # Danger alerts — interrupt immediately
TTS_PRIORITY_LOW  = 1       # Scene descriptions — plays in order

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — BLIP CAPTIONER
# ─────────────────────────────────────────────────────────────────────────────
BLIP_TRAIN_EPOCHS        = 8
BLIP_TRAIN_BATCH_SIZE    = 4         # Reduce to 2 if OOM on 4 GB VRAM
BLIP_LEARNING_RATE       = 5e-5
BLIP_WEIGHT_DECAY        = 0.01
BLIP_WARMUP_STEPS        = 500
BLIP_GRAD_ACCUM_STEPS    = 4         # Effective batch = 4 × 4 = 16
BLIP_SAVE_STEPS          = 500
BLIP_EVAL_STEPS          = 500

# Dataset mixing ratios (must sum to 1.0)
# Telugu IndicCaption gets highest weight — our primary output language
BLIP_TELUGU_RATIO        = 0.60      # AI4Bharat IndicCaption
BLIP_VIZWIZ_RATIO        = 0.25      # VizWiz (real blind-user photos, English)
# COCO fills the remaining 0.15 automatically
BLIP_MAX_TRAIN_SAMPLES   = None      # None = use full dataset
BLIP_MAX_VAL_SAMPLES     = 2000

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — YOLO DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
YOLO_TRAIN_EPOCHS        = 80        # More epochs for better accuracy
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
COLOR_DANGER_ZONE        = (0, 0, 200)
COLOR_TEXT_BG            = (0, 0, 0)
COLOR_TEXT_FG            = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = str(LOGS_DIR / "blind_project.log")
