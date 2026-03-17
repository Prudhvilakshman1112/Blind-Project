"""
src/vision_module.py
─────────────────────
Handles real-time camera capture, YOLO11s object detection, and
spatial analysis (clock-position, distance estimation, danger zone).

Trained model detects CAMPUS-SPECIFIC objects only (18 classes):
  person, bicycle, motorcycle, car, bench, chair, table, backpack,
  laptop, cell phone, door, openedDoor, window, stairs, step, ramp,
  pole, corridor

Three main classes:
  CameraStream    — Threaded OpenCV frame capture with a ring buffer
  ObjectDetector  — YOLO11 inference wrapper
  SpatialAnalyzer — Converts bounding boxes into human-readable positions

Danger Zone (tightened in 2025 overhaul):
  X-ratio = 0.28 (was 0.35), Y-ratio = 0.40 (was 0.50)
  This reduces false positives and focuses on true collision path.

High-Priority Objects (immediate audio interrupt):
  stairs, step, ramp, openedDoor, pole
  These always call tts.alert() — skipping the queue entirely.
"""

import time
import threading
import logging
from collections import deque
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FRAME_BUFFER_SIZE,
    YOLO_MODEL_NAME, YOLO_CUSTOM_WEIGHTS, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    YOLO_IMG_SIZE, YOLO_USE_CUSTOM,
    DANGER_ZONE_X_RATIO, DANGER_ZONE_Y_RATIO,
    DISTANCE_NEAR_THRESHOLD, DISTANCE_MEDIUM_THRESHOLD,
    COLOR_BBOX_DEFAULT, COLOR_BBOX_DANGER, COLOR_BBOX_HIGH_PRIORITY,
    COLOR_DANGER_ZONE, COLOR_TEXT_BG, COLOR_TEXT_FG,
    BBOX_THICKNESS, FONT_SCALE, DEVICE,
    HIGH_PRIORITY_OBJECTS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Detection Result type
# ─────────────────────────────────────────────────────────────────────────────
#
# Each detection is a dict with keys:
#   label (str), confidence (float), bbox (x1,y1,x2,y2),
#   clock_pos (str), distance_word (str), in_danger_zone (bool),
#   is_high_priority (bool)
#

DetectionList = List[Dict]


def is_high_priority(label: str) -> bool:
    """Returns True if the label requires an immediate audio interrupt."""
    return label.lower() in {obj.lower() for obj in HIGH_PRIORITY_OBJECTS}


# ─────────────────────────────────────────────────────────────────────────────
# CameraStream — Threaded frame capture
# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    """
    Captures frames from a webcam in a dedicated background thread.
    Consumers call .read() to get the latest frame without blocking.
    """

    def __init__(self, source: int | str = CAMERA_INDEX):
        self._source  = source
        self._buffer  = deque(maxlen=FRAME_BUFFER_SIZE)
        self._running = False
        self._thread  = None
        self._lock    = threading.Lock()
        self.frame_count = 0
        self.fps         = 0.0
        self._last_fps_time = time.time()

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self._source}\n"
                "Check CAMERA_INDEX in config.py or provide --source path."
            )

        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f"Camera stream started (source={self._source})")
        return self

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("Camera read failed — retrying …")
                time.sleep(0.05)
                continue
            with self._lock:
                self._buffer.append(frame)
            self.frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.fps = self.frame_count / (now - self._last_fps_time)
                self.frame_count    = 0
                self._last_fps_time = now

    def read(self) -> Optional[np.ndarray]:
        """Returns the latest frame or None if buffer is empty."""
        with self._lock:
            return self._buffer[-1].copy() if self._buffer else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self, "_cap"):
            self._cap.release()
        log.info("Camera stream stopped")


# ─────────────────────────────────────────────────────────────────────────────
# SpatialAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAnalyzer:
    """
    Converts bounding box coordinates into:
      - Clock position (12 o'clock = straight ahead, 3 = right, etc.)
      - Distance word (very close / nearby / in the distance)
      - Danger zone flag (bbox overlaps with tightened centre danger zone)

    Danger zone tuning (2025):
      X = 28% of frame width (centre band — tighter than previous 35%)
      Y = 40% of frame height (centre band — tighter than previous 50%)
      This reduces false alerts from objects at the periphery.
    """

    def __init__(self, frame_w: int = CAMERA_WIDTH, frame_h: int = CAMERA_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h

    def _bbox_center(self, x1, y1, x2, y2) -> Tuple[float, float]:
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def clock_position(self, x1, y1, x2, y2) -> str:
        """
        Maps bounding box centre to a clock position string.
        Frame grid:
            10 | 12 | 2
             9 | -- | 3
             8 |  6 | 4
        """
        cx, cy = self._bbox_center(x1, y1, x2, y2)
        nx = cx / self.frame_w
        ny = cy / self.frame_h

        if ny < 0.33:
            if nx < 0.33:    return "10 o'clock"
            elif nx < 0.67:  return "12 o'clock"
            else:            return "2 o'clock"
        elif ny < 0.67:
            if nx < 0.33:    return "9 o'clock"
            elif nx < 0.67:  return "directly ahead"
            else:            return "3 o'clock"
        else:
            if nx < 0.33:    return "8 o'clock"
            elif nx < 0.67:  return "6 o'clock"
            else:            return "4 o'clock"

    def distance_word(self, x1, y1, x2, y2) -> str:
        """Estimates distance from bounding box area as % of frame area."""
        bbox_area  = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_w * self.frame_h
        ratio      = bbox_area / frame_area

        if ratio >= DISTANCE_NEAR_THRESHOLD:
            return "very close"
        elif ratio >= DISTANCE_MEDIUM_THRESHOLD:
            return "nearby"
        else:
            return "in the distance"

    def in_danger_zone(self, x1, y1, x2, y2) -> bool:
        """
        Returns True if the bounding box overlaps the tightened centre danger zone.
        Danger zone = centre 28% × 40% of the frame (tighter than previous 35% × 50%).
        """
        dz_x1 = self.frame_w * (0.5 - DANGER_ZONE_X_RATIO / 2)
        dz_x2 = self.frame_w * (0.5 + DANGER_ZONE_X_RATIO / 2)
        dz_y1 = self.frame_h * (0.5 - DANGER_ZONE_Y_RATIO / 2)
        dz_y2 = self.frame_h * (0.5 + DANGER_ZONE_Y_RATIO / 2)

        ix1 = max(x1, dz_x1);  iy1 = max(y1, dz_y1)
        ix2 = min(x2, dz_x2);  iy2 = min(y2, dz_y2)
        return (ix2 > ix1) and (iy2 > iy1)

    def get_danger_zone_rect(self) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) pixel coords of the danger zone rectangle."""
        x1 = int(self.frame_w * (0.5 - DANGER_ZONE_X_RATIO / 2))
        x2 = int(self.frame_w * (0.5 + DANGER_ZONE_X_RATIO / 2))
        y1 = int(self.frame_h * (0.5 - DANGER_ZONE_Y_RATIO / 2))
        y2 = int(self.frame_h * (0.5 + DANGER_ZONE_Y_RATIO / 2))
        return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────────────────
# ObjectDetector
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Wraps YOLO11 for real-time campus object detection.
    Returns a list of enriched detection dicts (with spatial analysis).

    After campus fine-tuning, set YOLO_USE_CUSTOM = True in config.py to
    load the fine-tuned weights from checkpoints/yolo11_campus.pt.
    """

    def __init__(self):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        weights = YOLO_CUSTOM_WEIGHTS if YOLO_USE_CUSTOM else YOLO_MODEL_NAME
        log.info(f"Loading YOLO: {weights} on device={DEVICE}")
        self.model   = YOLO(weights)
        self.spatial = None
        log.info("YOLO model loaded ✓")

    def detect(self, frame: np.ndarray) -> DetectionList:
        """
        Runs YOLO inference on a frame.
        Returns list of dicts with detection info + spatial metadata.
        High-priority objects (stairs, doors, poles) are flagged for immediate alert.
        """
        h, w = frame.shape[:2]
        if self.spatial is None:
            self.spatial = SpatialAnalyzer(frame_w=w, frame_h=h)

        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU_THRESHOLD,
            imgsz=YOLO_IMG_SIZE,
            verbose=False,
        )

        detections: DetectionList = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf            = float(box.conf[0])
                cls_id          = int(box.cls[0])
                label           = result.names[cls_id]

                clock  = self.spatial.clock_position(x1, y1, x2, y2)
                dist   = self.spatial.distance_word(x1, y1, x2, y2)
                danger = self.spatial.in_danger_zone(x1, y1, x2, y2)
                hi_pri = is_high_priority(label)

                detections.append({
                    "label":            label,
                    "confidence":       round(conf, 2),
                    "bbox":             (x1, y1, x2, y2),
                    "clock_pos":        clock,
                    "distance_word":    dist,
                    "in_danger_zone":   danger,
                    "is_high_priority": hi_pri,
                })

        # Sort: high-priority first, then danger zone, then confidence
        detections.sort(key=lambda d: (
            not d["is_high_priority"],
            not d["in_danger_zone"],
            -d["confidence"]
        ))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Developer Window Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_dev_overlay(
    frame: np.ndarray,
    detections: DetectionList,
    fps: float,
    caption: str = "",
    reading_mode: bool = False,
) -> np.ndarray:
    """
    Draws bounding boxes, labels, clock positions, danger zone, FPS,
    and current caption onto the developer window frame.

    Colour coding:
      Green  = normal detection
      Red    = object in danger zone
      Orange = high-priority object (stairs, door, pole)
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Danger zone rectangle
    analyzer = SpatialAnalyzer(frame_w=w, frame_h=h)
    dz = analyzer.get_danger_zone_rect()
    cv2.rectangle(vis, (dz[0], dz[1]), (dz[2], dz[3]), COLOR_DANGER_ZONE, 2)
    cv2.putText(vis, "DANGER ZONE", (dz[0] + 4, dz[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DANGER_ZONE, 1)

    # Bounding boxes with priority-aware colouring
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["is_high_priority"]:
            color = COLOR_BBOX_HIGH_PRIORITY   # Orange — stairs/door/pole
        elif det["in_danger_zone"]:
            color = COLOR_BBOX_DANGER          # Red — in danger zone
        else:
            color = COLOR_BBOX_DEFAULT         # Green — normal
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        label_text = (
            f"{det['label']} {det['confidence']:.0%} | "
            f"{det['clock_pos']} | {det['distance_word']}"
        )
        if det["is_high_priority"]:
            label_text = "⚠ " + label_text
        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
        )
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_TEXT_BG, -1)
        cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT_FG, 1)

    # FPS counter
    cv2.putText(vis, f"FPS: {fps:.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Reading mode indicator
    if reading_mode:
        cv2.putText(vis, "[READING MODE]", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # Caption overlay (bottom)
    if caption:
        cap_text = f"Scene: {caption}"
        (cw, ch), _ = cv2.getTextSize(cap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (0, h - ch - 14), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, cap_text, (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return vis
