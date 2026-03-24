"""
data/download_datasets.py
─────────────────────────
Downloads / prepares all datasets needed for Blind-Project training.

DATASETS:
  1. Campus Caption Dataset  — YOUR OWN photos + Telugu captions
                               (see DATASET_CREATION_GUIDE.md to create this)
  2. Campus Detection Data   — Roboflow indoor/campus datasets
                               (requires manual download — see MANUAL_DOWNLOADS.md)

WHY NO HuggingFace TELUGU DATASETS:
  All previously referenced HF Telugu image-caption datasets have been
  confirmed deleted or made private as of March 2026:
    ✗ Hardik15/telugu-image-captions   — 404 Deleted
    ✗ FutureBeeAI/telugu-image-captions — 404 Deleted
    ✗ ai4bharat/IndicCOCO              — 404 Deleted
    ✗ Telugu-LLM-Labs/*               — Private/unavailable

  Since mBLIP already knows Telugu natively, we only need YOUR OWN
  campus photos (300-500 images) with Telugu captions written by you.
  This is better anyway — it teaches mBLIP your specific campus environment.

Usage:
    python data/download_datasets.py --dataset campus-setup
    python data/download_datasets.py --dataset manual-info
    python data/download_datasets.py --dataset verify
    python data/download_datasets.py --dataset all
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAMPUS_CAPTION_DIR, DATA_DIR, INDOOR_DIR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Campus Caption Dataset Setup
# Create the folder structure for your human-collected campus caption dataset.
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TRAIN = [
    {
        "idx": 0,
        "file_name": "train_000000.jpg",
        "caption": "ఒక వ్యక్తి నడవలో నడుస్తున్నాడు, ముందు మెట్లు కనిపిస్తున్నాయి."
    },
    {
        "idx": 1,
        "file_name": "train_000001.jpg",
        "caption": "ఒక విద్యార్థి బ్యాగు భుజాన వేసుకుని తరగతి గది తలుపు దగ్గర నిలబడ్డాడు."
    }
]

SAMPLE_VAL = [
    {
        "idx": 0,
        "file_name": "val_000000.jpg",
        "caption": "కాలేజ్ ఆవరణలో బెంచీలు మరియు చెట్లు ఉన్నాయి."
    }
]


def setup_campus_dataset() -> None:
    """
    Creates the folder structure for the human campus caption dataset
    and places sample JSON files showing the required format.

    After running this, follow DATASET_CREATION_GUIDE.md to fill in your
    own photos and Telugu captions.
    """
    log.info("=" * 65)
    log.info("  Setting up Campus Caption Dataset folder structure …")
    log.info("=" * 65)

    CAMPUS_CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = CAMPUS_CAPTION_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    train_json = CAMPUS_CAPTION_DIR / "train.json"
    val_json   = CAMPUS_CAPTION_DIR / "val.json"

    if train_json.exists() and val_json.exists():
        # Count existing entries
        with open(train_json, encoding="utf-8") as f:
            n_train = len(json.load(f))
        with open(val_json, encoding="utf-8") as f:
            n_val = len(json.load(f))
        log.info(f"Campus caption dataset already exists:")
        log.info(f"  train.json → {n_train} entries")
        log.info(f"  val.json   → {n_val} entries")
        log.info(f"  images/    → {CAMPUS_CAPTION_DIR / 'images'}")
        log.info("Nothing overwritten. Delete train.json / val.json to reset.")
        return

    # Write sample JSON files (format reference)
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_TRAIN, f, ensure_ascii=False, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_VAL, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("  ✓ Created folder structure:")
    log.info(f"    {CAMPUS_CAPTION_DIR}/")
    log.info(f"    ├── train.json  ← 2 sample entries (replace with your data)")
    log.info(f"    ├── val.json    ← 1 sample entry  (replace with your data)")
    log.info(f"    └── images/     ← place your .jpg campus photos here")
    log.info("")
    log.info("  NEXT STEP: Read DATASET_CREATION_GUIDE.md (in project root)")
    log.info("  It tells you exactly what photos to take and how to write Telugu captions.")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Manual Campus Detection Dataset Info
# ─────────────────────────────────────────────────────────────────────────────

def show_manual_info() -> None:
    """Prints information about campus YOLO datasets requiring manual download."""
    print("\n" + "=" * 65)
    print("  MANUAL DOWNLOADS — Campus Object Detection Datasets")
    print("  See full instructions: data/MANUAL_DOWNLOADS.md")
    print("=" * 65)
    print()
    print("  1. SCIN Indoor-Navigation-system (Roboflow)")
    print("     → https://universe.roboflow.com/scin/indoor-navigation-system")
    print("     → Classes: door, stairs (544 images)")
    print("     → Export as YOLOv11 → extract to: data/indoor_campus/scin_indoor/")
    print()
    print("  2. Akhash Indoor Navigation (Roboflow)")
    print("     → https://universe.roboflow.com/akhash/indoor-navigation")
    print("     → Classes: door, person, elevator, stair sign (1115 images)")
    print("     → Export as YOLOv11 → extract to: data/indoor_campus/akhash_indoor/")
    print()
    print("  3. Indoor Navigation for the Blind (Roboflow)")
    print("     → https://universe.roboflow.com")
    print("     → Search: 'IndoorNavigationForTheBlinds'")
    print("     → Classes: door, stairs, pole, chair, table")
    print("     → Export as YOLOv11 → extract to: data/indoor_campus/blind_indoor/")
    print()
    print("  4. Stairs Detection Dataset (Kaggle)")
    print("     → https://www.kaggle.com")
    print("     → Search: 'Stairs Detection YOLO Samuel Ayman'")
    print("     → 1000 annotated stair images in YOLO format")
    print("     → Extract to: data/indoor_campus/stairs_kaggle/")
    print()
    print("  After downloading, run:")
    print("     python training/train_detector.py --dataset campus")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Verify Downloads
# ─────────────────────────────────────────────────────────────────────────────

def verify_downloads() -> None:
    """Checks which datasets are present and reports status."""
    print("\n" + "=" * 60)
    print("  Dataset Verification Status")
    print("=" * 60)

    # Campus captions (mBLIP fine-tuning)
    train_ok = (CAMPUS_CAPTION_DIR / "train.json").exists()
    val_ok   = (CAMPUS_CAPTION_DIR / "val.json").exists()
    if train_ok and val_ok:
        with open(CAMPUS_CAPTION_DIR / "train.json", encoding="utf-8") as f:
            n_train = len(json.load(f))
        with open(CAMPUS_CAPTION_DIR / "val.json", encoding="utf-8") as f:
            n_val = len(json.load(f))
        if n_train <= 2:
            print(f"  ⚠ Campus captions : only sample data ({n_train} train entries)")
            print(f"    → Follow DATASET_CREATION_GUIDE.md to add your own images")
        else:
            print(f"  ✓ Campus captions : {n_train} train + {n_val} val pairs")
    else:
        print("  ✗ Campus captions : NOT set up")
        print("    → Run: python data/download_datasets.py --dataset campus-setup")

    # Campus detection datasets (YOLO)
    if INDOOR_DIR.exists():
        subs = [d for d in INDOOR_DIR.iterdir() if d.is_dir()]
        if subs:
            print(f"  ✓ Campus detection: {len(subs)} sub-dataset(s) found")
            for s in subs:
                print(f"      → {s.name}")
        else:
            print("  ✗ Campus detection: folder empty")
            print("    → See: data/MANUAL_DOWNLOADS.md")
    else:
        print("  ✗ Campus detection: NOT downloaded")
        print("    → See: data/MANUAL_DOWNLOADS.md")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download / prepare datasets for Blind-Project (mBLIP Campus Navigation)"
    )
    parser.add_argument(
        "--dataset",
        choices=["campus-setup", "all", "manual-info", "verify"],
        default="all",
        help=(
            "Which dataset action to perform:\n"
            "  campus-setup — Create folder structure for your campus caption dataset\n"
            "  all          — Run campus-setup + show manual download info (default)\n"
            "  manual-info  — Show Roboflow/Kaggle download instructions for YOLO\n"
            "  verify       — Check which datasets are already present"
        ),
    )
    args = parser.parse_args()

    if args.dataset == "manual-info":
        show_manual_info()
    elif args.dataset == "verify":
        verify_downloads()
    elif args.dataset == "campus-setup":
        setup_campus_dataset()
    else:
        # all
        setup_campus_dataset()
        print()
        show_manual_info()

    log.info("Done. Run 'python data/download_datasets.py --dataset verify' to check status.")


if __name__ == "__main__":
    main()
