"""
data/download_datasets.py
─────────────────────────
Downloads all datasets needed for Blind-Project training.

DATASETS:
  1. VizWiz-Captions   — English captions from real blind users
  2. MS-COCO 2017      — English scene captions (grammar quality)
  3. IndicCaption (te) — Telugu captions from AI4Bharat (HuggingFace)

NOTE: Indoor campus datasets (Kaggle/Roboflow) must be downloaded manually.
      See: data/MANUAL_DOWNLOADS.md for step-by-step instructions.

Usage:
    python data/download_datasets.py --dataset all
    python data/download_datasets.py --dataset vizwiz
    python data/download_datasets.py --dataset coco
    python data/download_datasets.py --dataset indic
    python data/download_datasets.py --dataset manual-info
"""

import os
import sys
import json
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VIZWIZ_DIR, COCO_DIR, INDIC_DIR, DATA_DIR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset URLs
# ─────────────────────────────────────────────────────────────────────────────

VIZWIZ_URLS = {
    "train_annotations": "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations/train.json",
    "val_annotations":   "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations/val.json",
    "test_annotations":  "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations/test.json",
    "train_images": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "val_images":   "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "test_images":  "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip",
}

COCO_URLS = {
    "train_images":  "http://images.cocodataset.org/zips/train2017.zip",
    "val_images":    "http://images.cocodataset.org/zips/val2017.zip",
    "annotations":   "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# AI4Bharat IndicCaption — Telugu subset
# HuggingFace dataset: ai4bharat/IndicCOCO
INDIC_HF_DATASET  = "ai4bharat/IndicCOCO"
INDIC_LANGUAGE    = "te"   # Telugu


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest_path: Path, desc: str = "") -> None:
    """Stream-downloads a file with a progress bar. Skips if already done."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        log.info(f"Already exists, skipping: {dest_path.name}")
        return
    log.info(f"Downloading {desc or dest_path.name} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    log.info(f"Saved: {dest_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extracts a zip archive. Skips if already extracted."""
    marker = extract_to / f".extracted_{zip_path.stem}"
    if marker.exists():
        log.info(f"Already extracted: {zip_path.name}")
        return
    log.info(f"Extracting {zip_path.name} → {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc=f"Extracting {zip_path.stem}"):
            zf.extract(member, extract_to)
    marker.touch()
    log.info(f"Extraction complete: {extract_to}")


# ─────────────────────────────────────────────────────────────────────────────
# VizWiz Download
# ─────────────────────────────────────────────────────────────────────────────

def download_vizwiz() -> None:
    """Downloads VizWiz-Captions (train / val / test) annotations + images."""
    log.info("=" * 60)
    log.info("Downloading VizWiz-Captions dataset …")
    log.info("  23,431 images taken by real blind users — English captions")
    log.info("=" * 60)

    ann_dir = VIZWIZ_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        download_file(VIZWIZ_URLS[f"{split}_annotations"], ann_dir / f"{split}.json",
                      desc=f"VizWiz {split} annotations")

    for split in ["train", "val", "test"]:
        zip_dest = VIZWIZ_DIR / f"{split}.zip"
        img_dest  = VIZWIZ_DIR / split
        download_file(VIZWIZ_URLS[f"{split}_images"], zip_dest, desc=f"VizWiz {split} images")
        extract_zip(zip_dest, img_dest)

    for split in ["train", "val"]:
        count = len(list((VIZWIZ_DIR / split).rglob("*.jpg")))
        log.info(f"VizWiz {split}: {count} images found")

    log.info("VizWiz download complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# COCO Download
# ─────────────────────────────────────────────────────────────────────────────

def download_coco() -> None:
    """Downloads MS-COCO 2017 train + val images and captions/annotations."""
    log.info("=" * 60)
    log.info("Downloading MS-COCO 2017 dataset …")
    log.info("  118k images — grammatically rich English captions")
    log.info("=" * 60)

    img_dir = COCO_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    train_zip = COCO_DIR / "train2017.zip"
    download_file(COCO_URLS["train_images"], train_zip, "COCO train2017 images (~18 GB)")
    extract_zip(train_zip, img_dir)

    val_zip = COCO_DIR / "val2017.zip"
    download_file(COCO_URLS["val_images"], val_zip, "COCO val2017 images (~1 GB)")
    extract_zip(val_zip, img_dir)

    ann_zip = COCO_DIR / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"], ann_zip, "COCO annotations (~241 MB)")
    extract_zip(ann_zip, COCO_DIR)

    cap_file = COCO_DIR / "annotations" / "captions_train2017.json"
    if cap_file.exists():
        with open(cap_file) as f:
            data = json.load(f)
        log.info(f"COCO train captions: {len(data['annotations'])} entries")
    else:
        log.warning("captions_train2017.json not found — check extraction")

    log.info("COCO 2017 download complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# IndicCaption Telugu Download  (AI4Bharat via HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def download_indic_caption() -> None:
    """
    Downloads AI4Bharat IndicCaption Telugu subset from HuggingFace.

    Dataset: ai4bharat/IndicCOCO
    Language: Telugu (te)
    Size: ~40k image-caption pairs in Telugu
    License: CC-BY-4.0

    Install requirement: pip install datasets huggingface-hub
    """
    log.info("=" * 60)
    log.info("Downloading AI4Bharat IndicCaption (Telugu) …")
    log.info("  ~40,000 COCO images with Telugu captions")
    log.info("  Dataset: ai4bharat/IndicCOCO on HuggingFace")
    log.info("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "HuggingFace 'datasets' library not installed.\n"
            "Run: pip install datasets huggingface-hub"
        )
        return

    INDIC_DIR.mkdir(parents=True, exist_ok=True)
    train_out = INDIC_DIR / "train_te.json"
    val_out   = INDIC_DIR / "val_te.json"

    if train_out.exists() and val_out.exists():
        log.info("IndicCaption Telugu already downloaded ✓")
        return

    log.info("Loading IndicCOCO — Telugu split from HuggingFace (may take a few minutes) …")
    try:
        # AI4Bharat IndicCOCO has language config splits
        ds_train = load_dataset(INDIC_HF_DATASET, INDIC_LANGUAGE, split="train", trust_remote_code=True)
        ds_val   = load_dataset(INDIC_HF_DATASET, INDIC_LANGUAGE, split="validation", trust_remote_code=True)

        # Save as JSON for our dataset_loader
        train_records = []
        for item in tqdm(ds_train, desc="Processing Telugu train captions"):
            train_records.append({
                "image_id":  item.get("image_id", ""),
                "file_name": item.get("file_name", ""),
                "caption":   item.get("caption", item.get("te", "")),
                "coco_url":  item.get("coco_url", ""),
            })

        val_records = []
        for item in tqdm(ds_val, desc="Processing Telugu val captions"):
            val_records.append({
                "image_id":  item.get("image_id", ""),
                "file_name": item.get("file_name", ""),
                "caption":   item.get("caption", item.get("te", "")),
                "coco_url":  item.get("coco_url", ""),
            })

        with open(train_out, "w", encoding="utf-8") as f:
            json.dump(train_records, f, ensure_ascii=False, indent=2)
        with open(val_out, "w", encoding="utf-8") as f:
            json.dump(val_records, f, ensure_ascii=False, indent=2)

        log.info(f"IndicCaption train: {len(train_records)} Telugu captions saved → {train_out}")
        log.info(f"IndicCaption val:   {len(val_records)} Telugu captions saved → {val_out}")
        log.info("IndicCaption Telugu download complete ✓")

    except Exception as e:
        log.error(f"IndicCaption download failed: {e}")
        log.error(
            "If the dataset config changed, try manually:\n"
            "  from datasets import load_dataset\n"
            f"  ds = load_dataset('{INDIC_HF_DATASET}', '{INDIC_LANGUAGE}')\n"
            "  Then save ds['train'] as JSON to data/indic_caption/train_te.json"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Manual Download Info
# ─────────────────────────────────────────────────────────────────────────────

def show_manual_info() -> None:
    """Prints information about datasets that require manual download."""
    print("\n" + "=" * 65)
    print("  MANUAL DOWNLOADS REQUIRED — Indoor Campus Datasets")
    print("=" * 65)
    print("  See full instructions: data/MANUAL_DOWNLOADS.md")
    print()
    print("  1. Indoor Objects Detection (Kaggle)")
    print("     → https://www.kaggle.com")
    print("     → Search: 'Indoor Objects Detection blind'")
    print("     → Download & extract to: data/indoor_campus/indoor_objects/")
    print()
    print("  2. Door + Stairs + Chairs Detection (Roboflow)")
    print("     → https://universe.roboflow.com")
    print("     → Search: 'door stairs chairs detection'")
    print("     → Export as YOLOv11 → extract to: data/indoor_campus/door_stairs/")
    print()
    print("  3. SmartCane Indoor Objects (Roboflow)")
    print("     → https://universe.roboflow.com")
    print("     → Search: 'smartcane indoor objects'")
    print("     → Export as YOLOv11 → extract to: data/indoor_campus/smartcane/")
    print()
    print("  After downloading, run:")
    print("     python training/train_detector.py --dataset combined")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Blind-Project (Telugu + Campus Navigation)"
    )
    parser.add_argument(
        "--dataset",
        choices=["vizwiz", "coco", "indic", "all", "manual-info"],
        default="all",
        help=(
            "Which dataset to download:\n"
            "  vizwiz      — VizWiz blind-user captions (English)\n"
            "  coco        — MS-COCO 2017 captions (English)\n"
            "  indic       — AI4Bharat IndicCaption Telugu captions\n"
            "  all         — All of the above (default)\n"
            "  manual-info — Show instructions for Kaggle/Roboflow manual downloads"
        ),
    )
    args = parser.parse_args()

    if args.dataset == "manual-info":
        show_manual_info()
        return

    if args.dataset in ("vizwiz", "all"):
        download_vizwiz()
    if args.dataset in ("coco", "all"):
        download_coco()
    if args.dataset in ("indic", "all"):
        download_indic_caption()

    log.info("\n✓ Downloads complete.")
    log.info(f"  VizWiz (English)  → {VIZWIZ_DIR}")
    log.info(f"  COCO   (English)  → {COCO_DIR}")
    log.info(f"  IndicCaption (te) → {INDIC_DIR}")
    log.info("\n  For indoor campus datasets (Kaggle/Roboflow):")
    log.info("     python data/download_datasets.py --dataset manual-info")


if __name__ == "__main__":
    main()
