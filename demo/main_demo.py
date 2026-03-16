"""
demo/main_demo.py
──────────────────
DEMO entry point — identical flow to main.py but uses:
  - GeminiCaptioner  instead of SceneCaptioner (BLIP)
  - Pre-trained YOLOv11n (no custom weights)
  - All other modules identical to the final app

Keyboard controls:
  Q — Quit
  R — Toggle OCR / Reading Mode
  P — Pause / resume TTS

Usage:
    python demo/main_demo.py
    python demo/main_demo.py --source 1          # different camera
    python demo/main_demo.py --source video.mp4  # test video
    python demo/main_demo.py --no-window         # headless
    python demo/main_demo.py --no-audio          # text output only
"""

import sys
import time
import argparse
import logging
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

# ── Make demo/ the first path so all imports resolve locally ──────────────────
DEMO_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(DEMO_DIR))

from config_demo import (
    DEV_WINDOW_TITLE, SHOW_DEV_WINDOW, LOG_LEVEL, LOG_FILE,
    CAPTION_INTERVAL, LOGS_DIR,
)
from vision_module      import CameraStream, ObjectDetector, draw_dev_overlay
from caption_module_demo import GeminiCaptioner, SpatialReasoningNLP
from audio_module       import TTSWorker
from ocr_module         import OCRReader

# ── Logging ──────────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("main_demo")


# ─────────────────────────────────────────────────────────────────────────────
# Demo Application
# ─────────────────────────────────────────────────────────────────────────────

class BlindNavigationDemo:
    """
    Demo application with identical UX to the final BlindNavigationApp.
    Only the captioning backend differs (Gemini instead of fine-tuned BLIP).
    """

    def __init__(self, args):
        self.args    = args
        self._running = False
        self._paused  = False

        self._last_caption    = ""
        self._last_detections = []
        self._caption_lock    = threading.Lock()
        self._detection_lock  = threading.Lock()

        log.info("=" * 55)
        log.info("  BLIND-PROJECT DEMO  (Gemini Flash + YOLOv11n)")
        log.info("=" * 55)
        log.info("Initialising modules …")

        self.camera    = CameraStream(source=args.source)
        self.detector  = ObjectDetector()
        self.captioner = GeminiCaptioner()
        self.reasoner  = SpatialReasoningNLP()
        self.ocr       = OCRReader()
        self.tts       = TTSWorker() if not args.no_audio else None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DemoWorker")

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        log.info("All modules ready ✓")

    # ── Startup / Shutdown ───────────────────────────────────────────────────

    def start(self):
        self._running = True
        self.camera.start()
        if self.tts:
            self.tts.start()
            self.tts.speak(
                "Blind navigation demo starting. "
                "Using Gemini A.I. for scene descriptions."
            )
        log.info("Demo running. Press Q to quit, R to toggle reading mode, P to pause.")
        self._main_loop()

    def stop(self):
        log.info("Shutting down demo …")
        self._running = False
        self._executor.shutdown(wait=False)
        self.camera.stop()
        if self.tts:
            self.tts.stop()
        cv2.destroyAllWindows()
        log.info("Demo stopped.")

    def _signal_handler(self, sig, frame):
        self.stop()
        sys.exit(0)

    # ── Caption Worker (async) ───────────────────────────────────────────────

    def _caption_worker(self, frame):
        try:
            raw = self.captioner.caption(frame)
            if raw is None:
                return   # Interval not elapsed

            with self._detection_lock:
                dets = list(self._last_detections)

            description = self.reasoner.build_description(raw, dets)

            with self._caption_lock:
                self._last_caption = raw

            log.info(f"[DESC] {description}")
            if self.tts and not self._paused:
                self.tts.speak(description)

        except Exception as e:
            log.error(f"Caption worker error: {e}", exc_info=True)

    # ── Main Loop ────────────────────────────────────────────────────────────

    def _main_loop(self):
        show_window = SHOW_DEV_WINDOW and not self.args.no_window
        prev_danger_labels = set()

        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Object detection (~30ms)
            detections = self.detector.detect(frame)
            with self._detection_lock:
                self._last_detections = detections

            # Danger zone — new objects trigger HIGH priority alert
            danger_items  = [d for d in detections if d["in_danger_zone"]]
            danger_labels = {d["label"] for d in danger_items}
            new_danger    = danger_labels - prev_danger_labels

            if new_danger and self.tts and not self._paused:
                top = next(d for d in danger_items if d["label"] in new_danger)
                alert = self.reasoner.build_danger_alert(top)
                self.tts.alert(alert)
                log.warning(f"[ALERT] {alert}")

            prev_danger_labels = danger_labels

            # BLIP replaced by Gemini (async, rate-limited by CAPTION_INTERVAL)
            self._executor.submit(self._caption_worker, frame.copy())

            # OCR / Reading Mode
            ocr_text = self.ocr.read_frame(frame)
            if ocr_text and self.tts:
                self.tts.speak(self.ocr.build_tts_message(ocr_text))
                log.info(f"[OCR] {ocr_text}")

            # Console fallback
            if self.args.no_audio and detections:
                labels = [d["label"] for d in detections[:3]]
                print(f"[DETECTIONS] {labels}")

            # Developer window
            if show_window:
                with self._caption_lock:
                    cap_text = self._last_caption

                vis = draw_dev_overlay(
                    frame,
                    detections,
                    fps=self.camera.fps,
                    caption=cap_text,
                    reading_mode=self.ocr.active,
                )
                cv2.imshow(DEV_WINDOW_TITLE, vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("Q pressed — quitting demo")
                    break
                elif key == ord("r"):
                    state = self.ocr.toggle()
                    if self.tts:
                        self.tts.speak("Reading mode on." if state else "Reading mode off.")
                elif key == ord("p"):
                    self._paused = not self._paused
                    log.info(f"TTS {'paused' if self._paused else 'resumed'}")
            else:
                time.sleep(0.001)

        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Blind-Project DEMO (Gemini Flash)")
    p.add_argument("--source",    default=0,
                   help="Camera index or video file path (default: 0)")
    p.add_argument("--no-window", action="store_true",
                   help="Headless mode — no developer window")
    p.add_argument("--no-audio",  action="store_true",
                   help="Disable TTS — print detections to console")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if isinstance(args.source, str) and args.source.isdigit():
        args.source = int(args.source)
    demo = BlindNavigationDemo(args)
    demo.start()
