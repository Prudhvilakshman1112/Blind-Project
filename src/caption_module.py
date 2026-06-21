"""
src/caption_module.py
──────────────────────
Handles image captioning using mBLIP (multilingual BLIP-2) and enriches
raw captions with spatial context from YOLO detections.

mBLIP UPGRADE (March 2026):
  - Replaced Salesforce/blip-image-captioning-base with Gregor/mblip-mt0-xl
  - mBLIP supports 96 languages including Telugu NATIVELY (no fine-tuning needed)
  - Uses MBLIP_PROMPT to instruct the model to respond in Telugu
  - Optional 4-bit quantization for RTX 3050 4 GB VRAM (MBLIP_USE_4BIT=True)
  - If MBLIP_USE_FINETUNED=True, loads LoRA adapter from checkpoints/mblip_campus

Two main classes:
  SceneCaptioner        — Loads mBLIP and generates raw Telugu captions
  SpatialReasoningNLP   — Combines mBLIP output with YOLO detections
                          to produce rich, actionable scene descriptions

Telugu output:
  mBLIP generates Telugu captions directly — no translation step needed.
  The audio_module speaks them via edge-tts (te-IN-ShrutiNeural).
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
    DEVICE, USE_FP16,
    MBLIP_PRETRAINED_NAME, MBLIP_FINETUNED_PATH,
    MBLIP_USE_FINETUNED, MBLIP_USE_4BIT,
    MBLIP_PROMPT, MBLIP_MAX_NEW_TOKENS, MBLIP_NUM_BEAMS,
    MBLIP_CAPTION_INTERVAL, HIGH_PRIORITY_OBJECTS,
    SCENE_CHANGE_THRESHOLD,
)
from src.vision_module import DetectionList

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SceneCaptioner
# ─────────────────────────────────────────────────────────────────────────────

class SceneCaptioner:
    """
    Loads mBLIP (Gregor/mblip-mt0-xl) and generates Telugu captions for
    input frames. Telugu is output natively — no translation needed.

    If MBLIP_USE_FINETUNED = True → loads LoRA adapter from
    checkpoints/mblip_campus/best/ to enhance campus-specific vocabulary.

    If MBLIP_USE_4BIT = True → uses 4-bit quantization to fit in 4 GB VRAM
    (RTX 3050). Set False for cloud training/deployment with 8+ GB VRAM.

    Respects MBLIP_CAPTION_INTERVAL to throttle inference — runs at most
    once every N seconds in the background thread.
    """

    def __init__(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        log.info(f"Loading mBLIP from: {MBLIP_PRETRAINED_NAME}")
        log.info(f"  4-bit quantization: {MBLIP_USE_4BIT}")
        log.info(f"  LoRA fine-tuned: {MBLIP_USE_FINETUNED}")

        self.processor = Blip2Processor.from_pretrained(MBLIP_PRETRAINED_NAME)

        # Load with optional 4-bit quantization
        if MBLIP_USE_4BIT:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    MBLIP_PRETRAINED_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                log.info("mBLIP loaded in 4-bit NF4 quantization ✓")
            except (ImportError, Exception) as e:
                log.warning(
                    f"4-bit quantization unavailable ({e}). "
                    "Falling back to float16. Needs ~8 GB VRAM."
                )
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    MBLIP_PRETRAINED_NAME,
                    torch_dtype=torch.float16,
                ).to(DEVICE)
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                MBLIP_PRETRAINED_NAME,
                torch_dtype=torch.float16,
            ).to(DEVICE)

        # Load LoRA adapter if campus fine-tuning complete
        if MBLIP_USE_FINETUNED:
            adapter_path = str(Path(MBLIP_FINETUNED_PATH) / "best")
            if Path(adapter_path).exists():
                try:
                    from peft import PeftModel
                    self.model.language_model = PeftModel.from_pretrained(
                        self.model.language_model, adapter_path
                    )
                    log.info(f"LoRA campus adapter loaded from: {adapter_path} ✓")
                except ImportError:
                    log.warning(
                        "PEFT not installed — cannot load LoRA adapter.\n"
                        "Run: pip install peft>=0.10.0"
                    )
                except Exception as e:
                    log.warning(f"Failed to load LoRA adapter: {e}")
            else:
                log.warning(
                    f"Campus fine-tuned adapter not found at: {adapter_path}\n"
                    "Using base mBLIP (still generates Telugu — just no campus tuning).\n"
                    "Run training/train_captioner.py first, then set MBLIP_USE_FINETUNED=True."
                )

        self.model.eval()
        self._last_caption_time = 0.0
        self._last_caption      = ""
        self._last_frame_gray   = None
        log.info("mBLIP captioner ready ✓")

    @torch.no_grad()
    def caption(self, frame: np.ndarray) -> Optional[str]:
        """
        Generates a Telugu caption for the given frame.
        Returns None if called before MBLIP_CAPTION_INTERVAL has elapsed.
        """
        now = time.time()
        if now - self._last_caption_time < MBLIP_CAPTION_INTERVAL:
            return None

        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Technique 5: Scene change detection
        if self._last_frame_gray is not None:
            diff = cv2.absdiff(self._last_frame_gray, gray)
            change = diff.mean()
            if change < SCENE_CHANGE_THRESHOLD:
                # Scene hasn't changed much, reuse last caption to save time
                self._last_caption_time = now
                return self._last_caption

        self._last_frame_gray = gray

        # Technique 3: Reduce image resolution
        small_frame = cv2.resize(frame, (224, 224))
        image  = Image.fromarray(cv2_to_pil(small_frame))
        
        inputs = self.processor(
            images=image,
            text=MBLIP_PROMPT,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.amp.autocast(
            device_type="cuda" if DEVICE == "cuda" else "cpu",
            enabled=USE_FP16
        ):
            ids = self.model.generate(
                **inputs,
                max_new_tokens=MBLIP_MAX_NEW_TOKENS,
                num_beams=MBLIP_NUM_BEAMS,
                do_sample=False,
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
    Enriches a raw mBLIP Telugu caption with spatial object information from
    YOLO to produce a complete, actionable scene description for the TTS engine.

    Output example:
        "మీరు ఒక నడవలో ఉన్నారు. హెచ్చరిక: మెట్లు చాలా దగ్గరగా ఉన్నాయి!"
        (You are in a corridor. Warning: stairs are very close!)

    Note: Since mBLIP outputs Telugu natively, the raw_caption from SceneCaptioner
    is already in Telugu. We append spatial info in English/Telugu as needed.
    """

    def build_description(
        self,
        raw_caption: str,
        detections: DetectionList,
    ) -> str:
        """
        Combines mBLIP Telugu caption + YOLO spatial detections into one message.
        High-priority objects (stairs, doors, poles) are mentioned immediately.

        Returns:
            Full descriptive sentence for the TTS engine.
        """
        parts: List[str] = []

        # 1. Scene-level context from mBLIP (already in Telugu)
        if raw_caption:
            scene = raw_caption.rstrip(".")
            parts.append(f"{scene}.")

        # 2. High-priority objects first (stairs/doors/poles — safety critical)
        high_pri_items = [d for d in detections if d.get("is_high_priority")]
        if high_pri_items:
            top = high_pri_items[0]
            parts.append(
                f"హెచ్చరిక: {top['label']} "
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
                f"జాగ్రత్త: {top['label']} "
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
                f"{det['label']} {det['distance_word']}, {det['clock_pos']}."
            )
            seen_labels.add(det["label"])
            mentioned += 1

        if not parts:
            return "మార్గం స్పష్టంగా ఉంది. అడ్డంకులు లేవు."  # Path is clear. No obstacles.

        return " ".join(parts)

    def build_danger_alert(self, detection: Dict) -> str:
        """
        Generates an urgent short alert for a danger-zone or high-priority object.
        Used for HIGH-priority TTS interrupts — plays immediately.
        """
        label = detection["label"]
        dist  = detection["distance_word"]
        pos   = detection["clock_pos"]

        if detection.get("is_high_priority"):
            return f"హెచ్చరిక! {label} {dist}, {pos}! జాగ్రత్తగా ఉండండి!"
        return f"జాగ్రత్త! {label} {dist}, {pos}!"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VOWELS = set("aeiouAEIOU")

def _article(word: str) -> str:
    """Returns 'an' if word starts with a vowel, else 'a'."""
    return "an" if word and word[0] in _VOWELS else "a"
