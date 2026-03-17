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
               (BLIP fine-tuned on HuggingFace Telugu captions)
    Output  — Telugu voice via Microsoft Neural TTS (edge-tts)
               or offline pyttsx3 fallback

  Dataset strategy (2025):
    YOLO  → Roboflow campus-specific datasets (door/stairs/pole …)
    BLIP  → Hardik15/telugu-image-captions  (HuggingFace, ~25K pairs)
    NO full COCO, NO VizWiz, NO IndicCOCO (all removed — too large
    or unavailable; irrelevant objects hurt campus accuracy)
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

# Caption dataset for BLIP fine-tuning (Telugu)
CAMPUS_CAPTION_DIR  = DATA_DIR / "telugu_captions"   # HuggingFace: Hardik15/telugu-image-captions

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
# BLIP CAPTIONING MODEL
# ─────────────────────────────────────────────────────────────────────────────
BLIP_PRETRAINED_NAME   = "Salesforce/blip-image-captioning-base"
BLIP_FINETUNED_PATH    = str(CHECKPOINTS_DIR / "blip_telugu")
BLIP_USE_FINETUNED     = False     # Set True after Telugu fine-tuning

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
# NOTE: edge-tts requires an active internet connection.
#       Latency: 800–1500ms per utterance (network dependent).
#       For offline use, pyttsx3 backend is used (English only).
#       Set TTS_ENGINE = "pyttsx3" for fully offline English operation.
TTS_ENGINE        = "edge-tts"            # "edge-tts" or "pyttsx3"
TTS_RATE          = 175
TTS_VOLUME        = 1.0
TTS_VOICE_GENDER  = "female"
EDGE_TTS_VOICE    = "te-IN-ShrutiNeural"  # Telugu female neural voice (Microsoft)

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
BLIP_WARMUP_STEPS        = 300
BLIP_GRAD_ACCUM_STEPS    = 4         # Effective batch = 4 × 4 = 16
BLIP_SAVE_STEPS          = 200
BLIP_EVAL_STEPS          = 200

# Training sample limits (prevents OOM on small GPUs)
BLIP_MAX_TRAIN_SAMPLES   = 20000     # Use up to 20K Telugu pairs (full dataset ~25K)
BLIP_MAX_VAL_SAMPLES     = 2000

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — YOLO DETECTOR
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
