"""
src/ocr_module.py
──────────────────
OCR "Reading Mode" using EasyOCR.

MODES:
  1. Manual   — User presses 'R' in the developer window
  2. Auto     — Activated automatically when YOLO detects a sign/board/notice
                with confidence ≥ OCR_AUTO_TRIGGER_CONFIDENCE (config.py)

Both Telugu (te) and English (en) are recognised simultaneously.
Detected text is sent to the HIGH-priority TTS queue for immediate reading.
"""

import time
import logging
from typing import List, Optional, Set

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OCR_LANGUAGES, OCR_GPU, OCR_CONFIDENCE,
    OCR_AUTO_TRIGGER_CLASSES, OCR_AUTO_TRIGGER_CONFIDENCE,
)

log = logging.getLogger(__name__)

_OCR_INTERVAL      = 2.0   # Seconds between OCR inference calls
_AUTO_COOLDOWN     = 5.0   # Seconds before auto-trigger can fire again


class OCRReader:
    """
    Wraps EasyOCR for bilingual (Telugu + English) text detection.

    Usage:
        reader = OCRReader()

        # Manual toggle (R key):
        reader.toggle()

        # Auto-trigger from YOLO detection labels:
        text = reader.read_frame(frame, detected_labels={"sign", "board"})
        if text:
            tts_worker.alert(f"Sign reads: {text}")
    """

    def __init__(self):
        log.info("Initialising EasyOCR … (first run downloads models ~100 MB)")
        import easyocr
        self.reader = easyocr.Reader(
            OCR_LANGUAGES,
            gpu=OCR_GPU,
            verbose=False,
        )
        self._last_read_time   = 0.0
        self._last_auto_time   = 0.0
        self._last_text        = ""
        self.active            = False   # Manual Reading Mode state
        log.info(f"EasyOCR initialised ✓  (GPU={OCR_GPU}, langs={OCR_LANGUAGES})")

    def toggle(self) -> bool:
        """Toggles manual Reading Mode on/off. Returns new state."""
        self.active = not self.active
        state = "ON" if self.active else "OFF"
        log.info(f"Reading Mode: {state}")
        return self.active

    def read_frame(
        self,
        frame: np.ndarray,
        detected_labels: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Runs OCR on the frame when:
          - Manual mode is active (user pressed 'R'), OR
          - Auto-trigger: a sign/board/notice label was detected by YOLO
            with confidence ≥ OCR_AUTO_TRIGGER_CONFIDENCE

        Returns combined detected text string, or None if nothing new.

        Args:
            frame            : Current camera frame (BGR numpy array)
            detected_labels  : Set of YOLO label strings from current frame
        """
        now = time.time()

        # Check auto-trigger condition
        auto_triggered = False
        if detected_labels:
            trigger_match = detected_labels & {cls.lower() for cls in OCR_AUTO_TRIGGER_CLASSES}
            if trigger_match and (now - self._last_auto_time) >= _AUTO_COOLDOWN:
                auto_triggered = True
                log.info(f"OCR auto-triggered by: {trigger_match}")

        # Only run OCR if manual mode OR auto-triggered
        if not self.active and not auto_triggered:
            return None

        # Rate-limit OCR calls
        if now - self._last_read_time < _OCR_INTERVAL:
            return None

        try:
            results = self.reader.readtext(frame, detail=1)
        except Exception as e:
            log.error(f"OCR error: {e}")
            return None

        self._last_read_time = now
        if auto_triggered:
            self._last_auto_time = now

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
        return f"Sign reads: {detected_text}"
