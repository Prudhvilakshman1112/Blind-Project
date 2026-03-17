"""
src/caption_module.py
──────────────────────
Handles image captioning using BLIP and enriches raw captions with
spatial context from YOLO detections.

WHAT CHANGED (2025 overhaul):
  - PRIORITY_OBJECTS expanded with all new campus navigation classes
  - build_danger_alert() uses Telugu-friendly short phrasing
  - SpatialReasoningNLP now flags high-priority objects for TTS interrupt

Two main classes:
  SceneCaptioner        — Loads BLIP and generates raw captions
  SpatialReasoningNLP   — Combines BLIP output with YOLO detections
                          to produce rich, actionable scene descriptions

Telugu output:
  After BLIP fine-tuning (training/train_captioner.py), the model will
  generate captions in Telugu natively. The audio_module then speaks them
  directly via edge-tts (te-IN-ShrutiNeural) without extra translation lag.
"""

import time
import logging
from typing import List, Optional, Dict

import torch
from PIL import Image
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, USE_FP16, BLIP_PRETRAINED_NAME, BLIP_FINETUNED_PATH,
    BLIP_USE_FINETUNED, BLIP_MAX_NEW_TOKENS, BLIP_NUM_BEAMS,
    BLIP_CAPTION_INTERVAL, HIGH_PRIORITY_OBJECTS,
)
from src.vision_module import DetectionList

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SceneCaptioner
# ─────────────────────────────────────────────────────────────────────────────

class SceneCaptioner:
    """
    Loads the BLIP image captioning model (fine-tuned or pre-trained)
    and generates captions for input frames.

    After Telugu fine-tuning:
      BLIP_USE_FINETUNED = True  → loads from checkpoints/blip_telugu/best
      Caption output will be in Telugu natively (no translation needed).

    Respects BLIP_CAPTION_INTERVAL to throttle inference calls —
    captioning runs at most every N seconds in the background.
    """

    def __init__(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        if BLIP_USE_FINETUNED:
            model_path = str(Path(BLIP_FINETUNED_PATH) / "best")
            if not Path(model_path).exists():
                log.warning(
                    f"Fine-tuned BLIP not found at {model_path}. "
                    "Falling back to pre-trained weights.\n"
                    "Run training/train_captioner.py first."
                )
                model_path = BLIP_PRETRAINED_NAME
        else:
            model_path = BLIP_PRETRAINED_NAME

        log.info(f"Loading BLIP captioner from: {model_path}")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model     = BlipForConditionalGeneration.from_pretrained(model_path)
        self.model.eval().to(DEVICE)

        self._last_caption_time = 0.0
        self._last_caption      = ""
        log.info("BLIP captioner loaded ✓")

    @torch.no_grad()
    def caption(self, frame: np.ndarray) -> Optional[str]:
        """
        Generates a caption for the given frame.
        Returns None if called before BLIP_CAPTION_INTERVAL has elapsed.
        """
        now = time.time()
        if now - self._last_caption_time < BLIP_CAPTION_INTERVAL:
            return None

        image  = Image.fromarray(cv2_to_pil(frame))
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu",
                                 enabled=USE_FP16):
            ids = self.model.generate(
                **inputs,
                max_new_tokens=BLIP_MAX_NEW_TOKENS,
                num_beams=BLIP_NUM_BEAMS,
                early_stopping=True,
            )

        caption = self.processor.decode(ids[0], skip_special_tokens=True).strip()
        self._last_caption_time = now
        self._last_caption      = caption
        return caption

    @property
    def last_caption(self) -> str:
        return self._last_caption


def cv2_to_pil(frame: np.ndarray) -> np.ndarray:
    """Converts a BGR OpenCV frame to RGB numpy array for PIL."""
    return frame[:, :, ::-1]


# ─────────────────────────────────────────────────────────────────────────────
# Priority Object Sets
# ─────────────────────────────────────────────────────────────────────────────

# Objects always mentioned first in scene descriptions
PRIORITY_OBJECTS = {
    # People
    "person",
    # Vehicles
    "car", "motorcycle", "bicycle",
    # Critical navigation hazards — expanded for campus use (2025)
    "stairs", "step", "ramp", "pole",
    "door", "openedDoor", "cabinetDoor",
    "elevator",
    # Common furniture obstacles
    "chair", "bench", "table",
    # Safety
    "fire hydrant", "stop sign",
}


# ─────────────────────────────────────────────────────────────────────────────
# SpatialReasoningNLP
# ─────────────────────────────────────────────────────────────────────────────

class SpatialReasoningNLP:
    """
    Enriches a raw BLIP caption with spatial object information from YOLO
    to produce a complete, actionable scene description for the TTS engine.

    Output example (English before Telugu translation):
        "You are in a corridor. Warning: stairs are very close, directly ahead!
         A door is nearby at your 3 o'clock."
    """

    def build_description(
        self,
        raw_caption: str,
        detections: DetectionList,
    ) -> str:
        """
        Combines BLIP caption + YOLO spatial detections into one sentence.
        High-priority objects (stairs, doors, poles) are mentioned immediately.

        Returns:
            Full descriptive sentence for the TTS engine.
        """
        parts: List[str] = []

        # 1. Scene-level context from BLIP
        if raw_caption:
            scene = raw_caption.rstrip(".")
            parts.append(f"You are looking at: {scene}.")

        # 2. High-priority objects first (stairs/doors/poles — safety critical)
        high_pri_items = [d for d in detections if d.get("is_high_priority")]
        if high_pri_items:
            top = high_pri_items[0]
            parts.append(
                f"Warning: {top['label']} is "
                f"{top['distance_word']}, {top['clock_pos']}!"
            )

        # 3. Remaining danger-zone objects
        seen_labels = {d["label"] for d in high_pri_items}
        danger_items = [
            d for d in detections
            if d["in_danger_zone"] and d["label"] not in seen_labels
        ]
        if danger_items:
            top = danger_items[0]
            parts.append(
                f"Caution: {_article(top['label'])} {top['label']} is "
                f"{top['distance_word']}, {top['clock_pos']}."
            )
            seen_labels.add(top["label"])

        # 4. Remaining top detections (up to 2 more)
        mentioned = 0
        for det in detections:
            if mentioned >= 2:
                break
            if det["label"] in seen_labels or det.get("is_high_priority") or det["in_danger_zone"]:
                continue
            parts.append(
                f"{_article(det['label']).capitalize()} {det['label']} is "
                f"{det['distance_word']}, {det['clock_pos']}."
            )
            seen_labels.add(det["label"])
            mentioned += 1

        if not parts:
            return "Path is clear. No obstacles detected."

        return " ".join(parts)

    def build_danger_alert(self, detection: Dict) -> str:
        """
        Generates an urgent, short alert for a danger-zone or high-priority object.
        Used for HIGH-priority TTS interrupts — plays immediately.
        """
        label = detection["label"]
        dist  = detection["distance_word"]
        pos   = detection["clock_pos"]

        if detection.get("is_high_priority"):
            return f"Alert! {label} {dist}, {pos}! Be careful!"
        return (
            f"Caution! {_article(label).capitalize()} "
            f"{label} is {dist}, {pos}!"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VOWELS = set("aeiouAEIOU")

def _article(word: str) -> str:
    """Returns 'an' if word starts with a vowel, else 'a'."""
    return "an" if word and word[0] in _VOWELS else "a"
