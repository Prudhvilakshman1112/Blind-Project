"""
demo/vision_module.py
──────────────────────
IDENTICAL to src/vision_module.py — only the config import is changed
to point to demo/config_demo.py instead of the root config.py.
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
sys.path.insert(0, str(Path(__file__).parent))   # demo/ directory first
from config_demo import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FRAME_BUFFER_SIZE,
    YOLO_MODEL_NAME, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_IMG_SIZE,
    DANGER_ZONE_X_RATIO, DANGER_ZONE_Y_RATIO,
    DISTANCE_NEAR_THRESHOLD, DISTANCE_MEDIUM_THRESHOLD,
    COLOR_BBOX_DEFAULT, COLOR_BBOX_DANGER, COLOR_DANGER_ZONE,
    COLOR_TEXT_BG, COLOR_TEXT_FG, BBOX_THICKNESS, FONT_SCALE, DEVICE,
)

log = logging.getLogger(__name__)

DetectionList = List[Dict]


# ─────────────────────────────────────────────────────────────────────────────
# CameraStream
# ─────────────────────────────────────────────────────────────────────────────

class CameraStream:
    def __init__(self, source: int | str = CAMERA_INDEX):
        self._source  = source
        self._buffer  = deque(maxlen=FRAME_BUFFER_SIZE)
        self._running = False
        self._thread  = None
        self._lock    = threading.Lock()
        self.frame_count    = 0
        self.fps            = 0.0
        self._last_fps_time = time.time()

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self._source}")
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
                self.fps            = self.frame_count / (now - self._last_fps_time)
                self.frame_count    = 0
                self._last_fps_time = now

    def read(self) -> Optional[np.ndarray]:
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
    def __init__(self, frame_w: int = CAMERA_WIDTH, frame_h: int = CAMERA_HEIGHT):
        self.frame_w = frame_w
        self.frame_h = frame_h

    def _bbox_center(self, x1, y1, x2, y2) -> Tuple[float, float]:
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def clock_position(self, x1, y1, x2, y2) -> str:
        """Return a natural spatial direction instead of clock positions.
        This translates cleanly into Telugu and other languages."""
        cx, cy = self._bbox_center(x1, y1, x2, y2)
        nx = cx / self.frame_w   # 0 = far left, 1 = far right
        ny = cy / self.frame_h   # 0 = top, 1 = bottom

        # Vertical zone: top third = upper, middle = eye-level, bottom = lower
        if ny < 0.33:
            if nx < 0.33:   return "to your upper-left"
            elif nx < 0.67: return "above you"
            else:           return "to your upper-right"
        elif ny < 0.67:
            if nx < 0.33:   return "to your left"
            elif nx < 0.67: return "directly ahead of you"
            else:           return "to your right"
        else:
            if nx < 0.33:   return "to your lower-left"
            elif nx < 0.67: return "ahead and below"
            else:           return "to your lower-right"

    def distance_word(self, x1, y1, x2, y2) -> str:
        """Return an approximate metre-based distance estimate.
        Based on the bounding-box area ratio relative to the full frame."""
        ratio = ((x2 - x1) * (y2 - y1)) / (self.frame_w * self.frame_h)
        if ratio >= DISTANCE_NEAR_THRESHOLD:
            return "about 1 metre away"
        elif ratio >= DISTANCE_MEDIUM_THRESHOLD:
            return "about 3 metres away"
        else:
            return "about 8 metres away"

    def in_danger_zone(self, x1, y1, x2, y2) -> bool:
        dz_x1 = self.frame_w * (0.5 - DANGER_ZONE_X_RATIO / 2)
        dz_x2 = self.frame_w * (0.5 + DANGER_ZONE_X_RATIO / 2)
        dz_y1 = self.frame_h * (0.5 - DANGER_ZONE_Y_RATIO / 2)
        dz_y2 = self.frame_h * (0.5 + DANGER_ZONE_Y_RATIO / 2)
        ix1 = max(x1, dz_x1); iy1 = max(y1, dz_y1)
        ix2 = min(x2, dz_x2); iy2 = min(y2, dz_y2)
        return (ix2 > ix1) and (iy2 > iy1)

    def get_danger_zone_rect(self) -> Tuple[int, int, int, int]:
        x1 = int(self.frame_w * (0.5 - DANGER_ZONE_X_RATIO / 2))
        x2 = int(self.frame_w * (0.5 + DANGER_ZONE_X_RATIO / 2))
        y1 = int(self.frame_h * (0.5 - DANGER_ZONE_Y_RATIO / 2))
        y2 = int(self.frame_h * (0.5 + DANGER_ZONE_Y_RATIO / 2))
        return x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────────────────
# ObjectDetector  (uses pretrained YOLOv11n — no custom weights needed)
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    def __init__(self):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        log.info(f"Loading YOLOv11n pre-trained: {YOLO_MODEL_NAME}  device={DEVICE}")
        self.model   = YOLO(YOLO_MODEL_NAME)   # downloads once, cached
        self.spatial = None
        log.info("YOLO model loaded ✓")

    def detect(self, frame: np.ndarray) -> DetectionList:
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
                conf   = float(box.conf[0])
                label  = result.names[int(box.cls[0])]
                detections.append({
                    "label":          label,
                    "confidence":     round(conf, 2),
                    "bbox":           (x1, y1, x2, y2),
                    "clock_pos":      self.spatial.clock_position(x1, y1, x2, y2),
                    "distance_word":  self.spatial.distance_word(x1, y1, x2, y2),
                    "in_danger_zone": self.spatial.in_danger_zone(x1, y1, x2, y2),
                })

        detections.sort(key=lambda d: (not d["in_danger_zone"], -d["confidence"]))
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Developer Window Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_dev_overlay(
    frame: np.ndarray,
    detections: DetectionList,
    fps: float,
    caption: str = "",
    reading_mode: bool = False,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Danger zone
    analyzer = SpatialAnalyzer(frame_w=w, frame_h=h)
    dz = analyzer.get_danger_zone_rect()
    cv2.rectangle(vis, (dz[0], dz[1]), (dz[2], dz[3]), COLOR_DANGER_ZONE, 2)
    cv2.putText(vis, "DANGER ZONE", (dz[0] + 4, dz[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DANGER_ZONE, 1)

    # Bounding boxes + labels
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLOR_BBOX_DANGER if det["in_danger_zone"] else COLOR_BBOX_DEFAULT
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
        label_text = (
            f"{det['label']} {det['confidence']:.0%} | "
            f"{det['clock_pos']} | {det['distance_word']}"
        )
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_TEXT_BG, -1)
        cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT_FG, 1)

    # FPS
    cv2.putText(vis, f"FPS: {fps:.1f}  [DEMO MODE - Gemini Flash]", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    if reading_mode:
        cv2.putText(vis, "[READING MODE]", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    if caption:
        cap_text = f"Scene: {caption}"
        (cw, ch), _ = cv2.getTextSize(cap_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (0, h - ch - 14), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, cap_text, (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return vis
