"""
demo/ocr_module.py
───────────────────
IDENTICAL to src/ocr_module.py — only the config import is changed
to point to demo/config_demo.py.
"""

import time
import logging
from typing import List, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config_demo import OCR_LANGUAGES, OCR_GPU, OCR_CONFIDENCE

log = logging.getLogger(__name__)

_OCR_INTERVAL = 2.0


class OCRReader:
    def __init__(self):
        log.info("Initialising EasyOCR … (first run downloads models ~100 MB)")
        import easyocr
        self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU, verbose=False)
        self._last_read_time = 0.0
        self._last_text      = ""
        self.active          = False
        log.info(f"EasyOCR initialised ✓  (GPU={OCR_GPU}, langs={OCR_LANGUAGES})")

    def toggle(self) -> bool:
        self.active = not self.active
        log.info(f"Reading Mode: {'ON' if self.active else 'OFF'}")
        return self.active

    def read_frame(self, frame: np.ndarray) -> Optional[str]:
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
        texts: List[str] = [
            text.strip()
            for (_, text, conf) in results
            if conf >= OCR_CONFIDENCE and text.strip()
        ]
        if not texts:
            return None
        combined = " | ".join(texts)
        if combined == self._last_text:
            return None
        self._last_text = combined
        return combined

    def build_tts_message(self, detected_text: str) -> str:
        return f"Text in view: {detected_text}"
