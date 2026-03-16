"""
src/ocr_module.py
──────────────────
OCR "Reading Mode" using EasyOCR.
Activated when the user presses 'R' in the developer window.

When active:
  1. Runs EasyOCR on the current frame every ~2 seconds.
  2. Filters out low-confidence detections.
  3. Sends combined text to HIGH-priority TTS queue.
"""

import time
import logging
from typing import List, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OCR_LANGUAGES, OCR_GPU, OCR_CONFIDENCE

log = logging.getLogger(__name__)

_OCR_INTERVAL = 2.0   # Seconds between OCR inference calls in Reading Mode


class OCRReader:
    """
    Wraps EasyOCR for text detection and reading.

    Usage:
        reader = OCRReader()
        text = reader.read_frame(frame)
        if text:
            tts_worker.speak(f"Text detected: {text}")
    """

    def __init__(self):
        log.info("Initialising EasyOCR … (first run downloads models ~100 MB)")
        import easyocr
        self.reader = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=OCR_GPU,
            verbose=False,
        )
        self._last_read_time = 0.0
        self._last_text      = ""
        self.active          = False
        log.info(f"EasyOCR initialised ✓  (GPU={OCR_GPU}, langs={OCR_LANGUAGES})")

    def toggle(self) -> bool:
        """Toggles Reading Mode on/off. Returns new state."""
        self.active = not self.active
        state = "ON" if self.active else "OFF"
        log.info(f"Reading Mode: {state}")
        return self.active

    def read_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Runs OCR on the frame if Reading Mode is active and interval has passed.
        Returns combined detected text string, or None if nothing new.
        """
        if not self.active:
            return None

        now = time.time()
        if now - self._last_read_time < _OCR_INTERVAL:
            return None

        try:
            results = self.reader.readtext(frame, detail=1)
        except Exception as e:
            log.error(f"OCR error: {e}")
            return None

        self._last_read_time = now

        # Filter by confidence and extract text
        texts: List[str] = [
            text.strip()
            for (_, text, confidence) in results
            if confidence >= OCR_CONFIDENCE and text.strip()
        ]

        if not texts:
            return None

        combined = " | ".join(texts)
        if combined == self._last_text:
            return None    # Don't repeat the same text

        self._last_text = combined
        return combined

    def build_tts_message(self, detected_text: str) -> str:
        """Formats the OCR result into a clear TTS sentence."""
        return f"Text in view: {detected_text}"
