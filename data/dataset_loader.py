"""
data/dataset_loader.py
──────────────────────
PyTorch Dataset classes for Blind-Project training.

CAPTION DATASET (for mBLIP LoRA fine-tuning):
  CampusCaptionDataset — Your own campus photos + Telugu captions
                         (see DATASET_CREATION_GUIDE.md)
                         Reads from data/campus_captions/train.json / val.json

  Uses Blip2Processor (for mBLIP / BLIP-2 architecture).

DETECTION DATASET (for YOLO training):
  Handled directly by the YOLO training script via YAML configs.
  CampusDetectionVerifier — verification helper only.

REMOVED (unavailable):
  ✗ TeluguCaptionDataset — used Hardik15/telugu-image-captions (deleted March 2026)
  ✗ IndicCaptionDataset  — ai4bharat/IndicCOCO no longer available
  ✗ VizWizCaptionDataset — noisy; not relevant to campus
  ✗ COCOCaptionDataset   — too large (18 GB); irrelevant classes
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CAMPUS_CAPTION_DIR, INDOOR_DIR,
    MBLIP_PRETRAINED_NAME,
    MBLIP_MAX_TRAIN_SAMPLES, MBLIP_MAX_VAL_SAMPLES,
    MBLIP_PROMPT,
    NUM_WORKERS,
)
from data.augmentations import get_train_transforms, get_val_transforms

import logging
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Campus Caption Dataset  (PRIMARY — your own photos + Telugu captions)
# ─────────────────────────────────────────────────────────────────────────────

class CampusCaptionDataset(Dataset):
    """
    Human-collected campus image-caption dataset in Telugu.

    Source  : Your own campus photos + Telugu captions written by native speakers
    Purpose : Fine-tune mBLIP with LoRA to improve campus-specific Telugu descriptions
    Format  : JSON file with list of {"idx", "file_name", "caption"} entries

    See DATASET_CREATION_GUIDE.md for instructions on creating this dataset.

    JSON format (data/campus_captions/train.json or val.json):
      [
        {
          "idx": 0,
          "file_name": "train_000000.jpg",
          "caption": "<Telugu caption text>"
        },
        ...
      ]

    Setup:
        python data/download_datasets.py --dataset campus-setup
    Then follow DATASET_CREATION_GUIDE.md to fill in images and captions.
    """

    def __init__(
        self,
        split: str = "train",
        processor=None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split     = split
        self.processor = processor
        self.augment   = augment
        self.transform = get_train_transforms() if augment else get_val_transforms()

        json_file = CAMPUS_CAPTION_DIR / f"{split}.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"Campus captions JSON not found: {json_file}\n"
                "Run setup first: python data/download_datasets.py --dataset campus-setup\n"
                "Then add your campus images and Telugu captions per DATASET_CREATION_GUIDE.md"
            )

        with open(json_file, encoding="utf-8") as f:
            records = json.load(f)

        self.img_dir = CAMPUS_CAPTION_DIR / "images"
        self.samples: List[Tuple[Optional[Path], str]] = []

        for rec in records:
            caption = rec.get("caption", "").strip()
            if not caption:
                continue
            fname = rec.get("file_name", "")
            img_path = self.img_dir / fname if fname else None
            self.samples.append((img_path, caption))

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        log.info(f"CampusCaptionDataset {split}: {len(self.samples):,} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, caption = self.samples[idx]

        # Load image — use a blank grey image if file is missing
        if img_path and img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            log.warning(f"Image not found: {img_path} — using blank grey placeholder")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        image = self.transform(image)

        if self.processor:
            # mBLIP uses prompt + caption together during training
            # The prompt tells the model to respond in Telugu
            encoding = self.processor(
                images=image,
                text=MBLIP_PROMPT,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=40,    # Prompt is short
            )

            # Encode the target caption separately as labels
            label_encoding = self.processor.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=80,    # Telugu output tokens
            )

            result = {k: v.squeeze(0) for k, v in encoding.items()}
            result["labels"] = label_encoding["input_ids"].squeeze(0)
            return result

        return {"image": image, "caption": caption}


# ─────────────────────────────────────────────────────────────────────────────
# Campus Detection Dataset Verifier
# ─────────────────────────────────────────────────────────────────────────────

class CampusDetectionVerifier:
    """
    Verifies that manual Roboflow campus detection datasets are in place.
    Does NOT load data — YOLO handles that via YAML configs.

    Usage:
        verifier = CampusDetectionVerifier()
        if verifier.is_ready():
            print("Ready to train YOLO")
        else:
            verifier.print_download_instructions()
    """

    def __init__(self):
        self.found_datasets = []
        if INDOOR_DIR.exists():
            self.found_datasets = [d for d in INDOOR_DIR.iterdir() if d.is_dir()]

    def is_ready(self) -> bool:
        return len(self.found_datasets) > 0

    def count_images(self) -> int:
        total = 0
        for ds_dir in self.found_datasets:
            total += len(list(ds_dir.rglob("*.jpg")))
            total += len(list(ds_dir.rglob("*.png")))
        return total

    def print_status(self):
        if self.is_ready():
            log.info(f"Campus detection datasets: {len(self.found_datasets)} found")
            for d in self.found_datasets:
                img_count = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
                log.info(f"  → {d.name}: {img_count} images")
        else:
            log.warning(
                "No campus detection datasets found!\n"
                f"Expected location: {INDOOR_DIR}\n"
                "See: data/MANUAL_DOWNLOADS.md for download instructions.\n"
                "Or run: python data/download_datasets.py --dataset manual-info"
            )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    processor,
    train_batch_size: int = 1,
    val_batch_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) ready for mBLIP LoRA fine-tuning.

    Uses CampusCaptionDataset — your own campus photos + Telugu captions.
    Follow DATASET_CREATION_GUIDE.md to prepare your dataset first.

    Note: Batch size defaults to 1 for RTX 3050 4 GB VRAM.
    Increase to 2–4 on Colab/Kaggle with 15+ GB VRAM.
    """
    train_ds = CampusCaptionDataset(
        split="train",
        processor=processor,
        augment=True,
        max_samples=MBLIP_MAX_TRAIN_SAMPLES,
    )
    val_ds = CampusCaptionDataset(
        split="val",
        processor=processor,
        augment=False,
        max_samples=MBLIP_MAX_VAL_SAMPLES,
    )

    log.info(f"DataLoaders ready — train: {len(train_ds):,} | val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(NUM_WORKERS > 0),
        drop_last=len(train_ds) > train_batch_size,  # Only drop if enough samples
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(NUM_WORKERS > 0),
    )

    return train_loader, val_loader
