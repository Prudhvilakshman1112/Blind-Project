"""
data/download_datasets.py
─────────────────────────
Downloads all datasets needed for Blind-Project training.

DATASETS:
  1. Telugu Image Captions  — Hardik15/telugu-image-captions (HuggingFace)
                               ~25,000 image-caption pairs in Telugu
                               Updated August 2024 — verified available
  2. Campus Detection Data  — Roboflow indoor/campus datasets
                               (requires manual download — see MANUAL_DOWNLOADS.md)

REMOVED (outdated / unavailable / too large):
  ✗ ai4bharat/IndicCOCO    — No longer available on HuggingFace
  ✗ VizWiz-Captions        — Noisy blind-user photos (irrelevant to campus)
  ✗ MS-COCO 2017 (~18 GB)  — Too large; irrelevant objects hurt campus accuracy

Usage:
    python data/download_datasets.py --dataset telugu
    python data/download_datasets.py --dataset manual-info
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
# Telugu Image Captions  (HuggingFace: Hardik15/telugu-image-captions)
# ─────────────────────────────────────────────────────────────────────────────

HF_TELUGU_DATASET = "Hardik15/telugu-image-captions"


def download_telugu_captions() -> None:
    """
    Downloads the Telugu image-caption dataset from HuggingFace.

    Dataset: Hardik15/telugu-image-captions
    Size   : ~25,000 image-caption pairs in Telugu
    License: CC-BY-4.0
    Updated: August 2024

    Saves:
      data/telugu_captions/train.json
      data/telugu_captions/val.json
      data/telugu_captions/images/      ← actual image files

    Install requirement: pip install datasets huggingface-hub Pillow
    """
    log.info("=" * 60)
    log.info("Downloading Telugu Image Captions (HuggingFace) …")
    log.info(f"  Dataset : {HF_TELUGU_DATASET}")
    log.info("  Size    : ~25,000 image-caption pairs in Telugu")
    log.info("=" * 60)

    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        import io
    except ImportError:
        log.error(
            "Required libraries not installed.\n"
            "Run: pip install datasets huggingface-hub Pillow"
        )
        return

    CAMPUS_CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    train_json = CAMPUS_CAPTION_DIR / "train.json"
    val_json   = CAMPUS_CAPTION_DIR / "val.json"
    img_dir    = CAMPUS_CAPTION_DIR / "images"

    if train_json.exists() and val_json.exists():
        log.info("Telugu captions already downloaded ✓")
        return

    img_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset from HuggingFace (first run may take a few minutes) …")
    try:
        ds = load_dataset(HF_TELUGU_DATASET, trust_remote_code=True)
    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        log.error(
            "\nTroubleshooting:\n"
            "  1. Ensure you have internet access\n"
            "  2. Run: huggingface-cli login  (if dataset requires auth)\n"
            f"  3. Manually browse: https://huggingface.co/datasets/{HF_TELUGU_DATASET}\n"
            "  4. If dataset unavailable, see MANUAL_DOWNLOADS.md for alternatives"
        )
        return

    # Detect available splits
    splits = list(ds.keys())
    log.info(f"Available splits: {splits}")

    # Use train split; if only one split, create 90/10 split
    if "train" in splits and "validation" in splits:
        train_split = ds["train"]
        val_split   = ds["validation"]
    elif "train" in splits and "test" in splits:
        train_split = ds["train"]
        val_split   = ds["test"]
    elif "train" in splits:
        full = ds["train"]
        split_ds = full.train_test_split(test_size=0.1, seed=42)
        train_split = split_ds["train"]
        val_split   = split_ds["test"]
    else:
        full = ds[splits[0]]
        split_ds = full.train_test_split(test_size=0.1, seed=42)
        train_split = split_ds["train"]
        val_split   = split_ds["test"]

    def _process_split(split_data, json_path: Path, split_name: str):
        records = []
        features = split_data.features
        log.info(f"Processing {split_name} ({len(split_data)} items) …")
        log.info(f"  Column names: {split_data.column_names}")

        # Detect column names flexibly
        caption_col = None
        for col in ["caption", "telugu_caption", "te_caption", "text", "label"]:
            if col in split_data.column_names:
                caption_col = col
                break
        if caption_col is None:
            caption_col = split_data.column_names[-1]
            log.warning(f"  Caption column not detected; using last column: '{caption_col}'")

        image_col = None
        for col in ["image", "img", "pixel_values"]:
            if col in split_data.column_names:
                image_col = col
                break

        for idx, item in enumerate(split_data):
            caption = str(item.get(caption_col, "")).strip()
            if not caption:
                continue

            img_filename = f"{split_name}_{idx:06d}.jpg"
            img_path = img_dir / img_filename

            # Save image if available in dataset
            if image_col and not img_path.exists():
                try:
                    img_data = item[image_col]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        pil_img = PILImage.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    elif hasattr(img_data, "save"):
                        pil_img = img_data.convert("RGB")
                    else:
                        pil_img = None

                    if pil_img:
                        pil_img.save(str(img_path), "JPEG", quality=90)
                except Exception:
                    img_path = None  # type: ignore

            records.append({
                "idx":       idx,
                "file_name": img_filename if img_path and img_path.exists() else "",
                "caption":   caption,
            })

            if idx % 1000 == 0:
                log.info(f"  → {idx}/{len(split_data)} processed …")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        log.info(f"  Saved {len(records)} records → {json_path}")
        return records

    train_records = _process_split(train_split, train_json, "train")
    val_records   = _process_split(val_split,   val_json,   "val")

    log.info("=" * 60)
    log.info(f"Telugu captions download complete ✓")
    log.info(f"  Train : {len(train_records):,} pairs → {train_json}")
    log.info(f"  Val   : {len(val_records):,} pairs  → {val_json}")
    log.info(f"  Images: {img_dir}")
    log.info("=" * 60)


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
    print("\n" + "=" * 50)
    print("  Dataset Verification Status")
    print("=" * 50)

    # Telugu captions
    train_ok = (CAMPUS_CAPTION_DIR / "train.json").exists()
    val_ok   = (CAMPUS_CAPTION_DIR / "val.json").exists()
    if train_ok and val_ok:
        import json as _json
        with open(CAMPUS_CAPTION_DIR / "train.json", encoding="utf-8") as f:
            n = len(_json.load(f))
        print(f"  ✓ Telugu captions : {n:,} train pairs")
    else:
        print("  ✗ Telugu captions : NOT downloaded")
        print("    → Run: python data/download_datasets.py --dataset telugu")

    # Campus detection datasets
    if INDOOR_DIR.exists():
        subs = [d for d in INDOOR_DIR.iterdir() if d.is_dir()]
        if subs:
            print(f"  ✓ Campus datasets : {len(subs)} sub-dataset(s) found")
            for s in subs:
                print(f"      → {s.name}")
        else:
            print("  ✗ Campus datasets : folder empty")
            print("    → See: data/MANUAL_DOWNLOADS.md")
    else:
        print("  ✗ Campus datasets : NOT downloaded")
        print("    → See: data/MANUAL_DOWNLOADS.md")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Blind-Project (Telugu Campus Navigation)"
    )
    parser.add_argument(
        "--dataset",
        choices=["telugu", "all", "manual-info", "verify"],
        default="all",
        help=(
            "Which dataset to download:\n"
            "  telugu      — Telugu image captions (HuggingFace, ~25K pairs)\n"
            "  all         — All auto-downloadable datasets (default)\n"
            "  manual-info — Show instructions for Roboflow/Kaggle manual downloads\n"
            "  verify      — Check which datasets are already present"
        ),
    )
    args = parser.parse_args()

    if args.dataset == "manual-info":
        show_manual_info()
    elif args.dataset == "verify":
        verify_downloads()
    else:
        # all or telugu
        if args.dataset in ("telugu", "all"):
            download_telugu_captions()
        if args.dataset == "all":
            print()
            show_manual_info()

    log.info("Done. Run 'python data/download_datasets.py --dataset verify' to check status.")


if __name__ == "__main__":
    main()
