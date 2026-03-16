"""
training/train_captioner.py
────────────────────────────
Fine-tunes Salesforce BLIP (blip-image-captioning-base) on the combined
VizWiz + MS-COCO dataset.  All computation is LOCAL — no API calls.

Key techniques used:
  - Gradient checkpointing  → fits 4 GB VRAM (RTX 3050)
  - Mixed precision (FP16)  → faster, lower VRAM
  - Gradient accumulation   → effective large batch size without OOM
  - Cosine LR schedule with warm-up
  - Best-checkpoint saving on val loss

Usage:
    python training/train_captioner.py
    python training/train_captioner.py --resume checkpoints/blip_finetuned/checkpoint_epoch3
    python training/train_captioner.py --epochs 5 --batch-size 2
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, USE_FP16, BLIP_PRETRAINED_NAME, BLIP_FINETUNED_PATH,
    BLIP_TRAIN_EPOCHS, BLIP_TRAIN_BATCH_SIZE, BLIP_LEARNING_RATE,
    BLIP_WEIGHT_DECAY, BLIP_WARMUP_STEPS, BLIP_GRAD_ACCUM_STEPS,
    BLIP_SAVE_STEPS, BLIP_EVAL_STEPS, LOGS_DIR,
)
from data.dataset_loader import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "blip_training.log")),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, epoch, accum_steps
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        input_ids       = batch["input_ids"].to(DEVICE)
        attention_mask  = batch["attention_mask"].to(DEVICE)
        pixel_values    = batch["pixel_values"].to(DEVICE)

        with autocast(enabled=USE_FP16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss / accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Validation", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values   = batch["pixel_values"].to(DEVICE)

        with autocast(enabled=USE_FP16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        total_loss += outputs.loss.item()
        pbar.set_postfix(val_loss=f"{total_loss / (pbar.n + 1):.4f}")

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    log.info(f"Using device: {DEVICE}  |  FP16: {USE_FP16}")
    save_dir = Path(BLIP_FINETUNED_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load processor and model
    log.info(f"Loading BLIP from: {BLIP_PRETRAINED_NAME} …")
    processor = BlipProcessor.from_pretrained(BLIP_PRETRAINED_NAME)
    model     = BlipForConditionalGeneration.from_pretrained(BLIP_PRETRAINED_NAME)

    # Enable gradient checkpointing to reduce VRAM usage
    model.gradient_checkpointing_enable()
    model.to(DEVICE)

    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            log.info(f"Resuming from checkpoint: {ckpt_path}")
            model = BlipForConditionalGeneration.from_pretrained(str(ckpt_path))
            model.to(DEVICE)
            start_epoch = int(ckpt_path.name.split("epoch")[-1]) + 1
        else:
            log.warning(f"Checkpoint not found: {ckpt_path}  — starting fresh")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        processor=processor,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size * 2,
    )

    # Optimizer and scheduler
    num_training_steps = (len(train_loader) // BLIP_GRAD_ACCUM_STEPS) * args.epochs
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=BLIP_WEIGHT_DECAY,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    scaler    = GradScaler(enabled=USE_FP16)

    log.info(f"Training steps: {num_training_steps}  |  Epochs: {args.epochs}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, BLIP_GRAD_ACCUM_STEPS,
        )
        val_loss = validate(model, val_loader)

        log.info(
            f"Epoch {epoch} — Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save epoch checkpoint
        epoch_dir = save_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(str(epoch_dir))
        processor.save_pretrained(str(epoch_dir))
        log.info(f"Checkpoint saved: {epoch_dir}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = save_dir / "best"
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            log.info(f"  ★ New best model saved (val_loss={best_val_loss:.4f})")

    log.info(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"  Best model → {save_dir / 'best'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP on VizWiz + COCO")
    parser.add_argument("--epochs",     type=int,   default=BLIP_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BLIP_TRAIN_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=BLIP_LEARNING_RATE)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    main(args)
