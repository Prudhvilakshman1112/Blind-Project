"""
demo/caption_module_demo.py
────────────────────────────
THE KEY DEMO DIFFERENCE:
  Instead of fine-tuned BLIP (which needs training), this module uses
  Google Gemini 1.5 Flash API to generate scene descriptions.

  The SpatialReasoningNLP class is IDENTICAL to the main project — it
  enriches the Gemini caption with YOLO detection positions.

  Output example:
    "You are in a living room. Warning! A chair is about 1 metre away,
     to your left. A person is about 3 metres away, to your right."
"""

import time
import logging
import io
from typing import List, Optional, Dict

import cv2
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config_demo import (
    GEMINI_API_KEY, GEMINI_MODEL, CAPTION_INTERVAL,
    CAPTION_MAX_TOKENS, GEMINI_PROMPT, SCENE_CHANGE_THRESHOLD
)

log = logging.getLogger(__name__)

DetectionList = List[Dict]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Captioner
# ─────────────────────────────────────────────────────────────────────────────

class GeminiCaptioner:
    """
    Sends a frame to Google Gemini 1.5 Flash and returns a scene description
    tailored for blind navigation.

    API limit: 15 requests/minute on free tier → we throttle to 1 per 4 s.
    """

    def __init__(self):
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "\n\n  !! GEMINI API KEY NOT SET !!\n"
                "  Open demo/config_demo.py and set GEMINI_API_KEY to your key.\n"
                "  Get a free key at: https://aistudio.google.com/apikey\n"
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            log.info(f"Gemini captioner initialised ✓  (model={GEMINI_MODEL})")
        except ImportError:
            raise ImportError(
                "google-genai not installed.\n"
                "Run: pip install google-genai"
            )

        self._last_caption_time = 0.0
        self._last_caption      = ""
        self._last_frame_gray   = None

    def caption(self, frame: np.ndarray) -> Optional[str]:
        """
        Sends frame to Gemini and returns a scene description.
        Returns None if CAPTION_INTERVAL has not elapsed yet.
        """
        now = time.time()
        if now - self._last_caption_time < CAPTION_INTERVAL:
            return None  # Too soon — skip call

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_frame_gray is not None:
            diff = cv2.absdiff(self._last_frame_gray, gray)
            if diff.mean() < SCENE_CHANGE_THRESHOLD:
                self._last_caption_time = now
                return self._last_caption
        
        self._last_frame_gray = gray

        # Convert OpenCV BGR frame to PIL image → JPEG bytes
        # Reduce image resolution for faster processing
        small_frame = cv2.resize(frame, (224, 224))
        rgb    = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buffer  = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()

        from google import genai
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    GEMINI_PROMPT,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=CAPTION_MAX_TOKENS,
                    temperature=0.3,
                ),
            )
            raw = response.text.strip()
            # Clean up any leading/trailing artefacts
            raw = raw.replace("\n", " ").strip()
            if not raw:
                return None

            self._last_caption_time = now
            self._last_caption      = raw
            log.info(f"[GEMINI] {raw[:120]}")
            return raw

        except Exception as e:
            log.error(f"Gemini API error: {e}")
            # Return last known caption so audio isn't silent on transient errors
            return self._last_caption or None

    @property
    def last_caption(self) -> str:
        return self._last_caption


# ─────────────────────────────────────────────────────────────────────────────
# SpatialReasoningNLP  (identical to caption_module.py in main project)
# ─────────────────────────────────────────────────────────────────────────────

_VOWELS = set("aeiouAEIOU")

def _article(word: str) -> str:
    return "an" if word and word[0] in _VOWELS else "a"


class SpatialReasoningNLP:
    """
    Enriches the Gemini raw caption with YOLO spatial detections to produce
    a complete, grammatically correct navigation sentence in Telugu.

    Identical logic to src/caption_module.py::SpatialReasoningNLP.
    """

    def build_description(
        self,
        raw_caption: str,
        detections: DetectionList,
    ) -> str:
        parts: List[str] = []

        # Scene-level context from Gemini
        if raw_caption:
            scene = raw_caption.rstrip(".")
            parts.append(f"{scene}.")

        # Priority: danger zone objects first
        danger_items = [d for d in detections if d["in_danger_zone"]]
        if danger_items:
            top = danger_items[0]
            parts.append(
                f"జాగ్రత్త: {top['label']} "
                f"{top['distance_word']}, {top['clock_pos']}."
            )

        # Up to 3 more notable detections
        seen = {d["label"] for d in danger_items}
        count = 0
        for det in detections:
            if count >= 3:
                break
            if det["in_danger_zone"] or det["label"] in seen:
                continue
            parts.append(
                f"{det['label']} {det['distance_word']}, {det['clock_pos']}."
            )
            seen.add(det["label"])
            count += 1

        return " ".join(parts) if parts else "మార్గం స్పష్టంగా ఉంది. అడ్డంకులు లేవు."

    def build_danger_alert(self, detection: Dict) -> str:
        label = detection["label"]
        dist  = detection["distance_word"]
        pos   = detection["clock_pos"]

        if detection.get("is_high_priority"):
            return f"హెచ్చరిక! {label} {dist}, {pos}! జాగ్రత్తగా ఉండండి!"
        return f"జాగ్రత్త! {label} {dist}, {pos}!"
