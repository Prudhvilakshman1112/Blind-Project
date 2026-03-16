"""
data/augmentations.py
──────────────────────
Image augmentation pipeline for training the BLIP captioning model.
Designed to simulate real-world camera conditions a blind user would face:
  - Motion blur (shaky hand)
  - Brightness/contrast shifts (indoor/outdoor lighting)
  - Random crops (partial scene views)
  - Gaussian noise (low-light grain)
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
from torchvision.transforms import functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Custom Transform: Gaussian Noise (PIL-based)
# ─────────────────────────────────────────────────────────────────────────────

class AddGaussianNoise:
    """Adds Gaussian noise to a PIL image to simulate low-light grain."""

    def __init__(self, mean: float = 0.0, std: float = 0.03):
        self.mean = mean
        self.std  = std

    def __call__(self, img: Image.Image) -> Image.Image:
        arr    = np.array(img).astype(np.float32) / 255.0
        noise  = np.random.normal(self.mean, self.std, arr.shape).astype(np.float32)
        noisy  = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255).astype(np.uint8))


class RandomMotionBlur:
    """Simulates camera motion blur by applying a box filter with some probability."""

    def __init__(self, radius: int = 2, p: float = 0.3):
        self.radius = radius
        self.p      = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return img.filter(ImageFilter.BoxBlur(self.radius))
        return img


class RandomBrightnessContrast:
    """Randomly adjusts brightness and contrast independently."""

    def __init__(
        self,
        brightness_range=(0.6, 1.4),
        contrast_range=(0.6, 1.4),
    ):
        self.brightness_range = brightness_range
        self.contrast_range   = contrast_range

    def __call__(self, img: Image.Image) -> Image.Image:
        b_factor = random.uniform(*self.brightness_range)
        c_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Transform Pipelines
# ─────────────────────────────────────────────────────────────────────────────

# BLIP expects 384×384 by default (base model)
_BLIP_IMG_SIZE = 384

# ImageNet normalisation statistics (BLIP uses these internally via processor,
# but we keep them here for sanity / direct PIL → tensor paths)
_MEAN = [0.48145466, 0.4578275,  0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]


def get_train_transforms() -> T.Compose:
    """
    Augmentation pipeline for training.
    Returns a torchvision Compose that takes a PIL image and returns a PIL image.
    (BLIP Processor does the final tensor conversion internally.)
    """
    return T.Compose([
        # Spatial augmentations
        T.RandomResizedCrop(
            size=_BLIP_IMG_SIZE,
            scale=(0.7, 1.0),       # Allow cropping to 70% of original
            ratio=(0.75, 1.33),
        ),
        T.RandomHorizontalFlip(p=0.5),

        # Colour / lighting augmentations (blind user conditions)
        RandomBrightnessContrast(
            brightness_range=(0.5, 1.5),
            contrast_range=(0.6, 1.4),
        ),
        T.ColorJitter(saturation=0.3, hue=0.05),

        # Blur & noise
        RandomMotionBlur(radius=2, p=0.25),
        AddGaussianNoise(mean=0.0, std=0.02),
    ])


def get_val_transforms() -> T.Compose:
    """Minimal pipeline for validation/inference — just resize."""
    return T.Compose([
        T.Resize((_BLIP_IMG_SIZE, _BLIP_IMG_SIZE)),
    ])
