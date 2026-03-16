"""
training/train_detector.py
───────────────────────────
Fine-tunes YOLOv11s on a COMBINED dataset:
  1. MS-COCO 2017       — 80 general object classes (base knowledge)
  2. Indoor campus data — door, stairs, chair, window, pole, etc.
                          (manually downloaded from Kaggle/Roboflow)
                          See: data/MANUAL_DOWNLOADS.md

Purpose:
  The final model must detect everything a blind person encounters
  inside a university or college campus — furniture, doors, stairs,
  poles, people, windows, and general objects.

Usage:
    python training/train_detector.py
    python training/train_detector.py --dataset combined   (default)
    python training/train_detector.py --dataset coco       (COCO only)
    python training/train_detector.py --dataset indoor     (indoor only)
    python training/train_detector.py --model yolo11s.pt --epochs 100
"""

import sys
import argparse
import yaml
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    COCO_DIR, INDOOR_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    YOLO_MODEL_NAME, YOLO_TRAIN_EPOCHS, YOLO_TRAIN_BATCH_SIZE,
    YOLO_TRAIN_IMG_SIZE, YOLO_TRAIN_LR0, YOLO_TRAIN_PATIENCE, DEVICE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "yolo_training.log")),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Class Lists
# ─────────────────────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Campus-specific classes from indoor datasets (Kaggle/Roboflow)
# These extend COCO with building/navigation-critical objects
CAMPUS_EXTRA_CLASSES = [
    "door", "openedDoor", "cabinetDoor", "window", "pole",
    "cabinet", "stairs", "ramp", "corridor",
]


# ─────────────────────────────────────────────────────────────────────────────
# YAML Generators
# ─────────────────────────────────────────────────────────────────────────────

def create_coco_yaml() -> Path:
    """Creates a YAML pointing to the local COCO dataset."""
    yaml_path = COCO_DIR / "coco.yaml"
    cfg = {
        "path":  str(COCO_DIR),
        "train": "images/train2017",
        "val":   "images/val2017",
        "nc":    80,
        "names": COCO_CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info(f"COCO YAML written: {yaml_path}")
    return yaml_path


def create_indoor_yaml() -> Path:
    """
    Creates a YAML pointing to the merged indoor campus dataset.
    It auto-discovers sub-folders inside INDOOR_DIR that contain
    images/train and images/val or train/val directly.
    """
    # Collect all image paths from subfolders
    train_dirs = []
    val_dirs   = []

    if not INDOOR_DIR.exists() or not any(INDOOR_DIR.iterdir()):
        log.warning(
            f"Indoor campus dataset directory is empty: {INDOOR_DIR}\n"
            "Please follow data/MANUAL_DOWNLOADS.md to download these datasets first."
        )
    else:
        for sub in INDOOR_DIR.iterdir():
            if not sub.is_dir():
                continue
            # Support both images/train and train/ structures
            for train_path in [sub / "images" / "train", sub / "train"]:
                if train_path.exists():
                    train_dirs.append(str(train_path))
                    break
            for val_path in [sub / "images" / "val", sub / "val"]:
                if val_path.exists():
                    val_dirs.append(str(val_path))
                    break

    log.info(f"Indoor datasets found: {len(train_dirs)} train splits")
    for d in train_dirs:
        log.info(f"  → {d}")

    all_classes = list(dict.fromkeys(COCO_CLASSES + CAMPUS_EXTRA_CLASSES))
    yaml_path   = INDOOR_DIR / "indoor_combined.yaml"
    cfg = {
        "path":  str(INDOOR_DIR),
        "train": train_dirs if train_dirs else ".",
        "val":   val_dirs   if val_dirs   else ".",
        "nc":    len(all_classes),
        "names": all_classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info(f"Indoor campus YAML written: {yaml_path}")
    return yaml_path


def create_combined_yaml() -> Path:
    """
    Creates a YAML that combines COCO + all indoor campus datasets.
    This is the recommended training configuration for the final app.
    """
    train_dirs: list = [str(COCO_DIR / "images" / "train2017")]
    val_dirs:   list = [str(COCO_DIR / "images" / "val2017")]

    if INDOOR_DIR.exists():
        for sub in INDOOR_DIR.iterdir():
            if not sub.is_dir():
                continue
            for train_path in [sub / "images" / "train", sub / "train"]:
                if train_path.exists():
                    train_dirs.append(str(train_path))
                    break
            for val_path in [sub / "images" / "val", sub / "val"]:
                if val_path.exists():
                    val_dirs.append(str(val_path))
                    break

    all_classes = list(dict.fromkeys(COCO_CLASSES + CAMPUS_EXTRA_CLASSES))
    yaml_path   = DATA_DIR_REF / "combined_detector.yaml"

    cfg = {
        "path":  str(COCO_DIR),
        "train": train_dirs,
        "val":   val_dirs,
        "nc":    len(all_classes),
        "names": all_classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info(f"Combined COCO+Indoor YAML written: {yaml_path}")
    log.info(f"  Total train splits : {len(train_dirs)}")
    log.info(f"  Total val splits   : {len(val_dirs)}")
    log.info(f"  Total classes      : {len(all_classes)}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    global DATA_DIR_REF
    DATA_DIR_REF = COCO_DIR.parent   # data/

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Select dataset configuration
    if args.dataset == "coco":
        yaml_path = create_coco_yaml()
        run_name  = "yolo_coco"
    elif args.dataset == "indoor":
        yaml_path = create_indoor_yaml()
        run_name  = "yolo_indoor_campus"
    else:   # combined (default)
        yaml_path = create_combined_yaml()
        run_name  = "yolo_combined_campus"

    log.info("=" * 60)
    log.info(f"  YOLOv11 Training — {run_name}")
    log.info(f"  Model  : {args.model}")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  Epochs : {args.epochs}")
    log.info(f"  Device : {DEVICE}")
    log.info("=" * 60)

    model    = YOLO(args.model)
    use_cuda = (DEVICE == "cuda")

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=YOLO_TRAIN_IMG_SIZE,
        batch=args.batch_size,
        lr0=YOLO_TRAIN_LR0,
        patience=YOLO_TRAIN_PATIENCE,
        device=0 if use_cuda else "cpu",
        project=str(LOGS_DIR / "yolo_runs"),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        amp=use_cuda,
        close_mosaic=10,
        verbose=True,
        plots=True,
    )

    # Copy best weights to checkpoints directory
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = CHECKPOINTS_DIR / "yolo11_custom.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, dest)
        log.info(f"✓ Best weights saved → {dest}")
    else:
        log.warning("best.pt not found in YOLO run directory")

    log.info(f"\n✓ YOLOv11 fine-tuning complete.")
    log.info(f"  Results         : {results.save_dir}")
    log.info(f"  Custom weights  : {dest}")
    log.info(f"  Next step: set YOLO_USE_CUSTOM = True in config.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv11 for College/Campus Blind Navigation"
    )
    parser.add_argument("--model",   type=str, default=YOLO_MODEL_NAME,
                        help="Base YOLO model file (default: yolo11s.pt)")
    parser.add_argument("--epochs",  type=int, default=YOLO_TRAIN_EPOCHS,
                        help=f"Training epochs (default: {YOLO_TRAIN_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=YOLO_TRAIN_BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--dataset", type=str,
                        choices=["coco", "indoor", "combined"],
                        default="combined",
                        help=(
                            "Dataset to train on:\n"
                            "  coco     — COCO 80 classes only\n"
                            "  indoor   — Indoor campus datasets only\n"
                            "  combined — COCO + indoor (recommended)"
                        ))
    args = parser.parse_args()
    main(args)
