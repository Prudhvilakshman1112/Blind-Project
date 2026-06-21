"""
demo/config_demo.py
────────────────────
Configuration for the DEMO version of Blind-Project.
Uses Google Gemini 1.5 Flash API for captioning instead of fine-tuned BLIP.
All other settings are identical to the main project.

SETUP: Replace YOUR_GEMINI_API_KEY_HERE with your free key from:
       https://aistudio.google.com/apikey
"""

import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# !! ACTION REQUIRED !!  Paste your free Gemini API key below:
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBV-LWbXga17lBioZdWrvo7nB4spxYYwqM"

GEMINI_MODEL   = "gemini-2.0-flash"   # Updated: gemini-1.5-flash no longer available

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEMO_DIR  = Path(__file__).parent.resolve()
ROOT_DIR  = DEMO_DIR.parent
LOGS_DIR  = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480
CAMERA_FPS        = 30
FRAME_BUFFER_SIZE = 5

# ─────────────────────────────────────────────────────────────────────────────
# API CONFIGURATION (Mobile Delivery)
# ─────────────────────────────────────────────────────────────────────────────
API_HOST          = "0.0.0.0"
API_PORT          = 8001
API_CORS_ORIGINS  = ["*"]


# ─────────────────────────────────────────────────────────────────────────────
# YOLO — Uses default pre-trained weights (no fine-tuning needed)
# ─────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_NAME      = "yolo11n.pt"   # Downloaded automatically on first run
YOLO_CONFIDENCE      = 0.45
YOLO_IOU_THRESHOLD   = 0.45
YOLO_IMG_SIZE        = 640

# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL REASONING
# ─────────────────────────────────────────────────────────────────────────────
DANGER_ZONE_X_RATIO       = 0.35
DANGER_ZONE_Y_RATIO       = 0.50
DISTANCE_NEAR_THRESHOLD   = 0.15
DISTANCE_MEDIUM_THRESHOLD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# CAPTIONING — Gemini Flash settings
# ─────────────────────────────────────────────────────────────────────────────
CAPTION_INTERVAL  = 4.0    # Seconds between Gemini API calls
CAPTION_MAX_TOKENS = 50    # Keep responses concise for TTS
SCENE_CHANGE_THRESHOLD = 15.0 # Skip caption if avg pixel diff is below this

# Prompt sent to Gemini with every frame (tuned for blind navigation)
GEMINI_PROMPT = (
    "You are an AI assistant helping a blind person navigate their environment. "
    "Describe what you see in this image in ONE clear, concise sentence in TELUGU. "
    "Focus on: the type of environment (indoor/outdoor), key objects, "
    "and anything that could be an obstacle or point of interest. "
    "Be specific and natural. "
    "Do NOT say 'The image shows' or 'I can see'. Just describe directly in Telugu."
)

# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES   = ["en"]
OCR_GPU         = (DEVICE == "cuda")
OCR_CONFIDENCE  = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
TTS_ENGINE        = "edge-tts"           # switched to edge-tts for Telugu support
TTS_RATE          = 175
TTS_VOLUME        = 1.0
TTS_VOICE_GENDER  = "female"
EDGE_TTS_VOICE    = "te-IN-ShrutiNeural" # Telugu female neural voice

# ─────────────────────────────────────────────────────────────────────────────
# TELUGU LANGUAGE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
TELUGU_MODE       = True    # Set False to revert to English voice
TTS_PRIORITY_HIGH = 0
TTS_PRIORITY_LOW  = 1

# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPER WINDOW
# ─────────────────────────────────────────────────────────────────────────────
DEV_WINDOW_TITLE    = "Blind-Project DEMO | Developer View"
SHOW_DEV_WINDOW     = True
BBOX_THICKNESS      = 2
FONT_SCALE          = 0.55
COLOR_BBOX_DEFAULT  = (0, 200, 0)
COLOR_BBOX_DANGER   = (0, 0, 255)
COLOR_DANGER_ZONE   = (0, 0, 200)
COLOR_TEXT_BG       = (0, 0, 0)
COLOR_TEXT_FG       = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = str(LOGS_DIR / "demo.log")
