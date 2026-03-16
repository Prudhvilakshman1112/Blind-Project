"""
data/dataset_loader.py
──────────────────────
PyTorch Dataset classes for all training data sources.

CAPTION DATASETS (used for BLIP fine-tuning):
  1. IndicCaptionDataset  — Telugu captions from AI4Bharat (PRIMARY - 60%)
  2. VizWizCaptionDataset — English captions by real blind users (25%)
  3. COCOCaptionDataset   — English COCO captions for grammar (15%)

The BLIP processor handles image resizing and tokenization internally.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from transformers import BlipProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    VIZWIZ_DIR, COCO_DIR, INDIC_DIR,
    BLIP_PRETRAINED_NAME, BLIP_VIZWIZ_RATIO, BLIP_TELUGU_RATIO,
    BLIP_MAX_TRAIN_SAMPLES, BLIP_MAX_VAL_SAMPLES,
    NUM_WORKERS,
)
from data.augmentations import get_train_transforms, get_val_transforms

import logging
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IndicCaption Telugu Dataset  (PRIMARY — AI4Bharat)
# ─────────────────────────────────────────────────────────────────────────────

class IndicCaptionDataset(Dataset):
    """
    AI4Bharat IndicCaption — Telugu (te) caption dataset.
    Images are the same COCO images; captions are in Telugu.

    JSON format (created by download_datasets.py --dataset indic):
      [ { "image_id": ..., "file_name": ..., "caption": "<Telugu text>",
          "coco_url": ... }, ... ]

    Images are loaded from COCO images directory (reuses COCO image files).
    Download: python data/download_datasets.py --dataset indic
    """

    def __init__(
        self,
        split: str = "train",
        processor: Optional[BlipProcessor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split     = split
        self.processor = processor
        self.augment   = augment
        self.transform = get_train_transforms() if augment else get_val_transforms()

        json_file = INDIC_DIR / f"{'train' if split == 'train' else 'val'}_te.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"IndicCaption Telugu JSON not found: {json_file}\n"
                "Run: python data/download_datasets.py --dataset indic"
            )

        with open(json_file, encoding="utf-8") as f:
            records = json.load(f)

        # COCO images are reused — point to the COCO images directory
        coco_split = "train2017" if split == "train" else "val2017"
        self.img_dir = COCO_DIR / "images" / coco_split

        # Build sample list
        self.samples: List[Tuple[Path, str]] = []
        for rec in records:
            caption = rec.get("caption", "").strip()
            fname   = rec.get("file_name", "")
            if not caption or not fname:
                continue
            img_path = self.img_dir / fname
            if img_path.exists():
                self.samples.append((img_path, caption))

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        log.info(f"IndicCaption Telugu {split}: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.processor:
            encoding = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=80,   # Telugu sentences may be slightly longer
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        return {"image": image, "caption": caption}


# ─────────────────────────────────────────────────────────────────────────────
# VizWiz Dataset  (English — real blind user photos)
# ─────────────────────────────────────────────────────────────────────────────

class VizWizCaptionDataset(Dataset):
    """
    VizWiz-Captions dataset — English captions of images taken by blind people.
    Each sample has one image and up to 5 crowd-sourced captions.
    We sample one caption randomly during training for diversity.

    Download: python data/download_datasets.py --dataset vizwiz
    """

    def __init__(
        self,
        split: str = "train",
        processor: Optional[BlipProcessor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split     = split
        self.processor = processor
        self.augment   = augment
        self.transform = get_train_transforms() if augment else get_val_transforms()

        ann_path = VIZWIZ_DIR / "annotations" / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"VizWiz annotations not found: {ann_path}\n"
                "Run: python data/download_datasets.py --dataset vizwiz"
            )

        with open(ann_path) as f:
            data = json.load(f)

        id2file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
        id2caps: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            cap = ann.get("caption", "").strip()
            if cap:
                id2caps.setdefault(ann["image_id"], []).append(cap)

        self.samples: List[Tuple[Path, List[str]]] = []
        img_dir = VIZWIZ_DIR / split
        for iid, caps in id2caps.items():
            img_path = img_dir / id2file[iid]
            if img_path.exists():
                self.samples.append((img_path, caps))

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        log.info(f"VizWiz {split}: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, captions = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = random.choice(captions) if self.augment else captions[0]

        if self.processor:
            encoding = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        return {"image": image, "caption": caption}


# ─────────────────────────────────────────────────────────────────────────────
# COCO Caption Dataset  (English — grammar quality)
# ─────────────────────────────────────────────────────────────────────────────

class COCOCaptionDataset(Dataset):
    """
    MS-COCO 2017 Captions — English, grammatically perfect captions.
    Used as a smaller training fraction to preserve sentence structure quality.

    Download: python data/download_datasets.py --dataset coco
    """

    def __init__(
        self,
        split: str = "train",
        processor: Optional[BlipProcessor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.split     = split
        self.processor = processor
        self.augment   = augment
        self.transform = get_train_transforms() if augment else get_val_transforms()

        ann_split = "train2017" if split == "train" else "val2017"
        ann_path  = COCO_DIR / "annotations" / f"captions_{ann_split}.json"
        self.img_dir = COCO_DIR / "images" / ann_split

        if not ann_path.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {ann_path}\n"
                "Run: python data/download_datasets.py --dataset coco"
            )

        with open(ann_path) as f:
            data = json.load(f)

        id2file: Dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}

        self.samples: List[Tuple[Path, str]] = []
        for ann in data["annotations"]:
            caption = ann.get("caption", "").strip()
            iid     = ann["image_id"]
            if not caption or iid not in id2file:
                continue
            img_path = self.img_dir / id2file[iid]
            if img_path.exists():
                self.samples.append((img_path, caption))

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        log.info(f"COCO {split}: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.processor:
            encoding = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        return {"image": image, "caption": caption}


# ─────────────────────────────────────────────────────────────────────────────
# Combined Caption Dataset  (Telugu-Primary Weighted Mix)
# ─────────────────────────────────────────────────────────────────────────────

class CombinedCaptionDataset(Dataset):
    """
    Combines Telugu IndicCaption + VizWiz + COCO with weighted sampling.

    Mixing ratios (from config.py):
      BLIP_TELUGU_RATIO = 0.60  →  IndicCaption Telugu (primary language)
      BLIP_VIZWIZ_RATIO = 0.25  →  VizWiz real blind-user photos
      COCO fills remaining 0.15 →  Grammatical English quality

    Why Telugu gets 60%:
      The final app output must be in Telugu. Training BLIP on Telugu
      captions makes it generate natively accurate Telugu descriptions.
    """

    def __init__(
        self,
        split: str = "train",
        processor: Optional[BlipProcessor] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        datasets = []

        # Telugu dataset (primary)
        try:
            indic = IndicCaptionDataset(split, processor, augment, max_samples)
            datasets.append(("indic", indic, BLIP_TELUGU_RATIO))
        except FileNotFoundError as e:
            log.warning(f"IndicCaption Telugu not available: {e}")
            log.warning("Proceeding without Telugu data. Run: python data/download_datasets.py --dataset indic")

        # VizWiz (blind-user photos, English)
        try:
            vizwiz = VizWizCaptionDataset(split, processor, augment, max_samples)
            datasets.append(("vizwiz", vizwiz, BLIP_VIZWIZ_RATIO))
        except FileNotFoundError as e:
            log.warning(f"VizWiz not available: {e}")

        # COCO (English grammar quality — remaining ratio)
        try:
            coco = COCOCaptionDataset(split, processor, augment, max_samples)
            total_other = sum(r for _, _, r in datasets)
            coco_ratio  = max(0.0, 1.0 - total_other)
            datasets.append(("coco", coco, coco_ratio))
        except FileNotFoundError as e:
            log.warning(f"COCO not available: {e}")

        if not datasets:
            raise RuntimeError("No datasets available! Run data/download_datasets.py first.")

        self._datasets = datasets
        self._flat     = []
        self._weights  = []

        for name, ds, ratio in datasets:
            n = len(ds)
            w = ratio / n if n > 0 else 0.0
            for i in range(n):
                self._flat.append((ds, i))
                self._weights.append(w)

        log.info(f"\nCombinedCaptionDataset ({split}):")
        for name, ds, ratio in datasets:
            log.info(f"  {name:12s}: {len(ds):>7,} samples  weight={ratio:.2f}")
        log.info(f"  {'TOTAL':12s}: {len(self._flat):>7,} samples")

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> Dict:
        ds, inner_idx = self._flat[idx]
        return ds[inner_idx]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self._weights,
            num_samples=len(self._weights),
            replacement=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    processor: BlipProcessor,
    train_batch_size: int = 4,
    val_batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) ready for BLIP fine-tuning.
    Uses weighted sampling to enforce Telugu priority ratio.
    """
    train_ds = CombinedCaptionDataset(
        split="train",
        processor=processor,
        augment=True,
        max_samples=BLIP_MAX_TRAIN_SAMPLES,
    )
    val_ds = CombinedCaptionDataset(
        split="val",
        processor=processor,
        augment=False,
        max_samples=BLIP_MAX_VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        sampler=train_ds.get_weighted_sampler(),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
