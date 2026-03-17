"""
main.py — Central Controller for Blind-Project
───────────────────────────────────────────────
Connects all modules: camera capture, YOLO detection, BLIP captioning,
OCR, and TTS. Runs on three concurrent threads using ThreadPoolExecutor.

2025 OVERHAUL — Key changes:
  • YOLO trained on 18 campus-specific classes only (no irrelevant COCO classes)
  • BLIP fine-tuned on Telugu captions — outputs Telugu directly
  • OCR auto-triggers when YOLO detects sign/board/notice (no 'R' key needed)
  • HIGH-PRIORITY objects (stairs, doors, poles) bypass TTS queue immediately
  • Tighter danger zone (28% × 40% of frame) for more precise obstacle detection

Keyboard controls (developer window):
  Q     — Quit application
  R     — Toggle OCR / Reading Mode manually
  P     — Pause / resume TTS descriptions

Usage:
    python main.py                      # Live webcam
    python main.py --source 0           # Explicit camera index
    python main.py --source video.mp4   # Test on a video file
    python main.py --no-window          # Headless (no dev window)
    python main.py --no-audio           # Disable TTS (text output only)
"""

import sys
import time
import argparse
import logging
import threading
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Set

import cv2
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DEV_WINDOW_TITLE, SHOW_DEV_WINDOW, LOG_LEVEL, LOG_FILE,
    BLIP_CAPTION_INTERVAL, LOGS_DIR,
    HIGH_PRIORITY_OBJECTS, OCR_AUTO_TRIGGER_CLASSES, OCR_AUTO_TRIGGER_CONFIDENCE,
)
from src.vision_module  import CameraStream, ObjectDetector, draw_dev_overlay
from src.caption_module import SceneCaptioner, SpatialReasoningNLP
from src.audio_module   import TTSWorker
from src.ocr_module     import OCRReader

# ── Logging setup ────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

class BlindNavigationApp:
    """
    Top-level application that orchestrates all modules.

    Thread layout:
        Main thread      — OpenCV window, keyboard events
        Thread 1 (cam)   — Camera capture  (CameraStream already threaded)
        Thread 2 (pool)  — BLIP captioning (slow, every 4 s)
        Thread 3 (TTS)   — Audio output    (TTSWorker has its own thread)

    Priority logic:
        HIGH-PRIORITY objects (stairs/doors/poles) → immediate tts.alert()
        Danger-zone objects (other)                → tts.alert() on new entry
        Scene descriptions                         → tts.speak() (queued)
        OCR text (sign auto-detected / manual)     → tts.alert() (interrupts)
    """

    def __init__(self, args):
        self.args      = args
        self._running  = False
        self._paused   = False

        # Shared state (read by multiple threads)
        self._last_caption     = ""
        self._last_detections  = []
        self._caption_lock     = threading.Lock()
        self._detection_lock   = threading.Lock()

        # Modules
        log.info("Initialising modules …")
        self.camera    = CameraStream(source=args.source)
        self.detector  = ObjectDetector()
        self.captioner = SceneCaptioner()
        self.reasoner  = SpatialReasoningNLP()
        self.ocr       = OCRReader()
        self.tts       = TTSWorker() if not args.no_audio else None

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="InfWorker")

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        log.info("All modules initialised ✓")

    # ─── Startup / Shutdown ──────────────────────────────────────────────────

    def start(self):
        self._running = True
        self.camera.start()
        if self.tts:
            self.tts.start()
            self.tts.speak("Campus navigation system ready.")

        log.info("Application running. Press Q to quit, R for reading mode, P to pause.")
        self._main_loop()

    def stop(self):
        log.info("Shutting down …")
        self._running = False
        self._executor.shutdown(wait=False)
        self.camera.stop()
        if self.tts:
            self.tts.stop()
        cv2.destroyAllWindows()
        log.info("Shutdown complete.")

    def _signal_handler(self, sig, frame):
        log.info("Signal received — stopping …")
        self.stop()
        sys.exit(0)

    # ─── Background Captioning Thread ────────────────────────────────────────

    def _caption_worker(self, frame: np.ndarray):
        """
        Runs BLIP captioning + spatial reasoning in the executor pool.
        Updates shared caption state and enqueues TTS.
        """
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
            log.error(f"Captioning worker error: {e}", exc_info=True)

    # ─── Main Loop ───────────────────────────────────────────────────────────

    def _main_loop(self):
        show_window        = SHOW_DEV_WINDOW and not self.args.no_window
        prev_danger_labels: Set[str] = set()
        prev_high_pri_labels: Set[str] = set()

        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Object Detection (synchronous — fast, ~30ms) ─────────────────
            detections = self.detector.detect(frame)
            with self._detection_lock:
                self._last_detections = detections

            # Build sets for alerting logic
            all_labels       = {d["label"] for d in detections}
            danger_labels    = {d["label"] for d in detections if d["in_danger_zone"]}
            high_pri_labels  = {d["label"] for d in detections if d.get("is_high_priority")}

            # ── HIGH-PRIORITY Alert (stairs/doors/poles — immediate) ─────────
            new_high_pri = high_pri_labels - prev_high_pri_labels
            if new_high_pri and self.tts and not self._paused:
                top = next(d for d in detections
                           if d["label"] in new_high_pri and d.get("is_high_priority"))
                alert = self.reasoner.build_danger_alert(top)
                self.tts.alert(alert)
                log.warning(f"[HIGH-PRI ALERT] {alert}")

            prev_high_pri_labels = high_pri_labels

            # ── Danger Zone Alerts (other objects in centre) ──────────────────
            new_danger = (danger_labels - high_pri_labels) - prev_danger_labels
            if new_danger and self.tts and not self._paused:
                top_danger = next(
                    d for d in detections
                    if d["label"] in new_danger and d["in_danger_zone"]
                )
                alert = self.reasoner.build_danger_alert(top_danger)
                self.tts.alert(alert)
                log.warning(f"[DANGER ALERT] {alert}")

            prev_danger_labels = danger_labels

            # ── BLIP Captioning (async — slow, every 4 s) ────────────────────
            self._executor.submit(self._caption_worker, frame.copy())

            # ── OCR — Auto-trigger from sign/board/notice detections ─────────
            # Auto-triggers when YOLO detects sign/board/notice class
            # Also runs when manual Reading Mode is active (R key)
            lowered_labels = {lbl.lower() for lbl in all_labels}
            ocr_text = self.ocr.read_frame(frame, detected_labels=lowered_labels)
            if ocr_text and self.tts:
                msg = self.ocr.build_tts_message(ocr_text)
                self.tts.alert(msg)   # OCR text is HIGH-priority — reads immediately
                log.info(f"[OCR] {ocr_text}")

            # ── Console fallback (no audio) ──────────────────────────────────
            if self.args.no_audio and detections:
                print(f"[DETECTIONS] {[d['label'] for d in detections[:3]]}")

            # ── Developer Window ─────────────────────────────────────────────
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
                    log.info("Q pressed — quitting")
                    break
                elif key == ord("r"):
                    state = self.ocr.toggle()
                    if self.tts:
                        mode_msg = "Reading mode on." if state else "Reading mode off."
                        self.tts.speak(mode_msg)
                elif key == ord("p"):
                    self._paused = not self._paused
                    state_word   = "paused" if self._paused else "resumed"
                    log.info(f"TTS {state_word}")
            else:
                time.sleep(0.001)

        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Blind-Project — Real-Time Campus Scene Understanding & Telugu Voice Navigation"
    )
    parser.add_argument(
        "--source", default=0,
        help="Camera index (int) or video file path. Default: 0 (webcam)",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Disable developer OpenCV window (headless mode)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable TTS (print detections to console instead)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert --source to int if digit string
    if isinstance(args.source, str) and args.source.isdigit():
        args.source = int(args.source)

    app = BlindNavigationApp(args)
    app.start()
