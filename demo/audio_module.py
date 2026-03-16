"""
demo/audio_module.py
─────────────────────
Same as src/audio_module.py with one addition:
  - TeluguTranslator: translates English → Telugu using deep-translator
  - When TELUGU_MODE=True, all spoken text is translated before being
    sent to edge-tts (te-IN-ShrutiNeural voice).
"""

import queue
import threading
import logging
import time
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config_demo import (
    TTS_ENGINE, TTS_RATE, TTS_VOLUME, TTS_VOICE_GENDER,
    EDGE_TTS_VOICE, TTS_PRIORITY_HIGH, TTS_PRIORITY_LOW,
    TELUGU_MODE,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Telugu Translator
# ─────────────────────────────────────────────────────────────────────────────

class TeluguTranslator:
    """
    Translates English text to Telugu using deep-translator (GoogleTranslator).
    Falls back to original English text on any error so TTS never goes silent.
    """

    def __init__(self):
        try:
            from deep_translator import GoogleTranslator
            self._translator = GoogleTranslator(source="en", target="te")
            log.info("Telugu translator initialised ✓  (deep-translator)")
        except ImportError:
            log.warning(
                "deep-translator not installed. Telugu translation disabled.\n"
                "Run: pip install deep-translator"
            )
            self._translator = None

    def translate(self, text: str) -> str:
        if not self._translator or not text:
            return text
        try:
            result = self._translator.translate(text)
            return result if result else text
        except Exception as e:
            log.warning(f"Translation error (falling back to English): {e}")
            return text


# Shared singleton translator — created once
_translator: Optional[TeluguTranslator] = None

def _get_translator() -> TeluguTranslator:
    global _translator
    if _translator is None:
        _translator = TeluguTranslator()
    return _translator


def maybe_translate(text: str) -> str:
    """Translate to Telugu if TELUGU_MODE is enabled."""
    if TELUGU_MODE:
        return _get_translator().translate(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# pyttsx3 Backend  (kept as English-only fallback)
# ─────────────────────────────────────────────────────────────────────────────

class Pyttsx3Backend:
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate",   TTS_RATE)
        self.engine.setProperty("volume", TTS_VOLUME)
        voices = self.engine.getProperty("voices")
        target = TTS_VOICE_GENDER.lower()
        for v in voices:
            if target in v.name.lower() or target in v.id.lower():
                self.engine.setProperty("voice", v.id)
                break
        self._lock = threading.Lock()
        log.info("pyttsx3 TTS backend initialised ✓")

    def speak(self, text: str):
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()

    def stop(self):
        self.engine.stop()


# ─────────────────────────────────────────────────────────────────────────────
# edge-tts Backend  (supports Telugu neural voice)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeTTSBackend:
    def __init__(self):
        self._fallback: Optional[Pyttsx3Backend] = None
        try:
            import edge_tts        # noqa
            import pygame
            pygame.mixer.init()
            self._pygame = pygame
            log.info(f"edge-tts backend initialised ✓  (voice={EDGE_TTS_VOICE})")
        except ImportError as e:
            log.warning(f"edge-tts/pygame unavailable ({e}). Falling back to pyttsx3.")
            self._fallback = Pyttsx3Backend()

    def speak(self, text: str):
        # Translate to Telugu before speaking
        text = maybe_translate(text)

        if self._fallback:
            self._fallback.speak(text)
            return

        import asyncio, tempfile, os, edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            await communicate.save(tmp)
            self._pygame.mixer.music.load(tmp)
            self._pygame.mixer.music.play()
            while self._pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)
            # Clean up temp file after playback
            try:
                os.unlink(tmp)
            except Exception:
                pass

        asyncio.run(_run())

    def stop(self):
        if self._fallback:
            self._fallback.stop()
        else:
            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Priority TTS Queue
# ─────────────────────────────────────────────────────────────────────────────

class PriorityTTSQueue:
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._seq   = 0

    def put(self, text: str, priority: int = TTS_PRIORITY_LOW):
        self._seq += 1
        self._queue.put((priority, self._seq, text))

    def get(self, timeout: float = 0.5):
        try:
            _, _, text = self._queue.get(timeout=timeout)
            return text
        except queue.Empty:
            return None

    def clear(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def empty(self) -> bool:
        return self._queue.empty()


# ─────────────────────────────────────────────────────────────────────────────
# TTSWorker
# ─────────────────────────────────────────────────────────────────────────────

class TTSWorker:
    def __init__(self):
        if TTS_ENGINE.lower() == "edge-tts":
            self._backend = EdgeTTSBackend()
        else:
            self._backend = Pyttsx3Backend()
        self._tts_queue = PriorityTTSQueue()
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._speaking  = False

    def start(self) -> "TTSWorker":
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker_loop, daemon=True, name="TTSWorkerThread"
        )
        self._thread.start()
        log.info("TTS worker started ✓")
        return self

    def _worker_loop(self):
        while self._running:
            text = self._tts_queue.get(timeout=0.3)
            if text:
                self._speaking = True
                try:
                    self._backend.speak(text)
                except Exception as e:
                    log.error(f"TTS error: {e}")
                finally:
                    self._speaking = False

    def speak(self, text: str):
        if text and text.strip():
            self._tts_queue.put(text.strip(), priority=TTS_PRIORITY_LOW)

    def alert(self, text: str):
        if text and text.strip():
            self._tts_queue.clear()
            self._tts_queue.put(text.strip(), priority=TTS_PRIORITY_HIGH)
            log.info(f"[TTS ALERT] {text[:80]}")

    def stop(self):
        self._running = False
        self._tts_queue.clear()
        self._backend.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
