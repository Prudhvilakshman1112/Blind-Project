"""
training/train_detector.py
───────────────────────────
Fine-tunes YOLO11s on campus-specific indoor datasets only.

WHY NOT FULL COCO:
  Full COCO (80 classes, 18 GB) is NOT used here. Training on irrelevant
  classes (elephant, pizza, frisbee …) degrades accuracy for campus objects
  and wastes GPU time. Our 18-class campus model is faster and more reliable.

CAMPUS CLASSES (18 total):
  People & mobility: person, bicycle, motorcycle, car
  Furniture: bench, chair, table, backpack, laptop, cell phone
  Navigation (CRITICAL): door, openedDoor, window, stairs, step, ramp, pole, corridor

DATASETS (Roboflow — manual download required):
  See: data/MANUAL_DOWNLOADS.md for exact links and instructions.

Usage:
    python training/train_detector.py                   (default: campus)
    python training/train_detector.py --dataset campus  (campus Roboflow only)
    python training/train_detector.py --model yolo11m.pt --epochs 100
"""

import sys
import argparse
import yaml
import shutil
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INDOOR_DIR, CHECKPOINTS_DIR, LOGS_DIR, DATA_DIR,
    YOLO_MODEL_NAME, YOLO_TRAIN_EPOCHS, YOLO_TRAIN_BATCH_SIZE,
    YOLO_TRAIN_IMG_SIZE, YOLO_TRAIN_LR0, YOLO_TRAIN_PATIENCE, DEVICE,
    CAMPUS_CLASSES,
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
# YAML Generator — Campus Datasets
# ─────────────────────────────────────────────────────────────────────────────

def create_campus_yaml() -> Path:
    """
    Creates a YAML config for YOLOv11 training using all campus Roboflow datasets.
    Auto-discovers sub-folders inside INDOOR_DIR that have images/train structure.

    Supports two common Roboflow export structures:
      images/train + images/val   (standard)
      train/images + valid/images (alternative)
    """
    train_dirs: list = []
    val_dirs:   list = []

    if not INDOOR_DIR.exists() or not any(INDOOR_DIR.iterdir()):
        log.error(
            f"No campus datasets found in: {INDOOR_DIR}\n"
            "Please download first — see: data/MANUAL_DOWNLOADS.md\n"
            "Or run: python data/download_datasets.py --dataset manual-info"
        )
        sys.exit(1)

    for sub in sorted(INDOOR_DIR.iterdir()):
        if not sub.is_dir():
            continue

        # Structure 1: images/train + images/valid (or images/val)
        for train_candidate in [sub / "images" / "train", sub / "train" / "images"]:
            if train_candidate.exists():
                train_dirs.append(str(train_candidate))
                break

        for val_candidate in [
            sub / "images" / "valid",
            sub / "images" / "val",
            sub / "valid" / "images",
            sub / "val" / "images",
        ]:
            if val_candidate.exists():
                val_dirs.append(str(val_candidate))
                break

    if not train_dirs:
        log.error(
            "No training image folders found inside campus datasets.\n"
            "Ensure Roboflow datasets are extracted correctly.\n"
            "Expected structure: data/indoor_campus/<name>/images/train/"
        )
        sys.exit(1)

    log.info(f"Campus training splits found: {len(train_dirs)}")
    for d in train_dirs:
        log.info(f"  train → {d}")
    for d in val_dirs:
        log.info(f"  val   → {d}")

    yaml_path = DATA_DIR / "campus_detector.yaml"
    cfg = {
        "path":  str(INDOOR_DIR),
        "train": train_dirs,
        "val":   val_dirs if val_dirs else train_dirs,   # Fall back to train if no val
        "nc":    len(CAMPUS_CLASSES),
        "names": CAMPUS_CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    log.info(f"Campus YAML written: {yaml_path}")
    log.info(f"  Classes ({len(CAMPUS_CLASSES)}): {CAMPUS_CLASSES}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    from data.dataset_loader import CampusDetectionVerifier
    verifier = CampusDetectionVerifier()
    verifier.print_status()
    if not verifier.is_ready():
        log.error("No campus datasets found. See data/MANUAL_DOWNLOADS.md")
        sys.exit(1)

    yaml_path = create_campus_yaml()
    run_name  = f"yolo11_campus_{args.model.replace('.pt', '')}"

    log.info("=" * 60)
    log.info(f"  YOLO11 Campus Training — {run_name}")
    log.info(f"  Model   : {args.model}")
    log.info(f"  Classes : {len(CAMPUS_CLASSES)} (campus-specific only)")
    log.info(f"  Epochs  : {args.epochs}")
    log.info(f"  Device  : {DEVICE}")
    log.info(f"  Batch   : {args.batch_size}")
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
    dest = CHECKPOINTS_DIR / "yolo11_campus.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, dest)
        log.info(f"✓ Best weights saved → {dest}")
    else:
        log.warning("best.pt not found in YOLO run directory")

    log.info(f"\n✓ YOLO11 campus fine-tuning complete.")
    log.info(f"  Run results     : {results.save_dir}")
    log.info(f"  Custom weights  : {dest}")
    log.info(f"  Next step: set YOLO_USE_CUSTOM = True in config.py")
    log.info(f"  And set  : YOLO_CUSTOM_WEIGHTS = '{dest}' in config.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11 for College/Campus Blind Navigation"
    )
    parser.add_argument(
        "--model", type=str, default=YOLO_MODEL_NAME,
        help="Base YOLO model file (default: yolo11s.pt). Use yolo11m.pt for +8%% mAP if 8GB+ VRAM."
    )
    parser.add_argument(
        "--epochs", type=int, default=YOLO_TRAIN_EPOCHS,
        help=f"Training epochs (default: {YOLO_TRAIN_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=YOLO_TRAIN_BATCH_SIZE,
        help="Batch size (default: 16; reduce to 8 if OOM)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["campus"], default="campus",
        help="Dataset mode (only 'campus' supported — Roboflow indoor datasets)"
    )
    args = parser.parse_args()
    main(args)
