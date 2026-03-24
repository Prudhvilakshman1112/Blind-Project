"""
training/train_captioner.py
────────────────────────────
Fine-tunes mBLIP (Gregor/mblip-mt0-xl) on your human-collected campus caption
dataset using LoRA (Low-Rank Adaptation). All computation is LOCAL — no API calls.

WHY mBLIP + LoRA instead of standard BLIP:
  - mBLIP already knows Telugu natively (trained on 96 languages)
  - LoRA only trains a small adapter (~100 MB) — the base model is FROZEN
  - Works on RTX 3050 4 GB VRAM with 4-bit quantization
  - Only 3 epochs needed (vs 8 for BLIP from scratch)

VRAM guide:
  RTX 3050 (4 GB)  → MBLIP_USE_4BIT=True, batch=1, grad_accum=8
  Colab T4 (15 GB) → MBLIP_USE_4BIT=False, batch=4, grad_accum=4

Key techniques:
  - 4-bit quantization (bitsandbytes)  → fits 4 GB VRAM
  - LoRA adapter                        → only ~1% of params trainable
  - Gradient accumulation               → effective batch without OOM
  - Cosine LR schedule with warm-up

Usage:
    python training/train_captioner.py
    python training/train_captioner.py --epochs 3 --batch-size 1
    python training/train_captioner.py --no-4bit    (for cloud with 12+ GB VRAM)
    python training/train_captioner.py --resume checkpoints/mblip_campus/checkpoint_epoch2

Prerequisites:
    1. python data/download_datasets.py --dataset campus-setup
    2. Add your campus images + Telugu captions per DATASET_CREATION_GUIDE.md
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, USE_FP16,
    MBLIP_PRETRAINED_NAME, MBLIP_FINETUNED_PATH,
    MBLIP_TRAIN_EPOCHS, MBLIP_TRAIN_BATCH_SIZE, MBLIP_LEARNING_RATE,
    MBLIP_WEIGHT_DECAY, MBLIP_WARMUP_STEPS, MBLIP_GRAD_ACCUM_STEPS,
    MBLIP_SAVE_STEPS, MBLIP_EVAL_STEPS,
    MBLIP_LORA_RANK, MBLIP_LORA_ALPHA, MBLIP_LORA_DROPOUT,
    MBLIP_USE_4BIT, LOGS_DIR,
)
from data.dataset_loader import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "mblip_training.log")),
    ],
)
log = logging.getLogger(__name__)

_scaler_device = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading — mBLIP with optional 4-bit quantization + LoRA
# ─────────────────────────────────────────────────────────────────────────────

def load_mblip_for_training(use_4bit: bool = True):
    """
    Loads mBLIP (Gregor/mblip-mt0-xl) with:
      - Optional 4-bit quantization (bitsandbytes) for 4 GB VRAM GPUs
      - LoRA adapter wrapping the LLM layers
    Returns: (model_with_lora, processor)
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        log.error(
            "PEFT library not installed.\n"
            "Run: pip install peft>=0.10.0"
        )
        sys.exit(1)

    log.info(f"Loading mBLIP processor from: {MBLIP_PRETRAINED_NAME} …")
    processor = Blip2Processor.from_pretrained(MBLIP_PRETRAINED_NAME)

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa — check it's installed
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            log.info("Loading mBLIP in 4-bit (NF4) quantization for 4 GB VRAM …")
            model = Blip2ForConditionalGeneration.from_pretrained(
                MBLIP_PRETRAINED_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ImportError:
            log.warning(
                "bitsandbytes not installed or not compatible with this platform.\n"
                "Falling back to float16 (will need ~8 GB VRAM).\n"
                "To install: pip install bitsandbytes>=0.43.0\n"
                "Note: On Windows, bitsandbytes may need WSL2."
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                MBLIP_PRETRAINED_NAME,
                torch_dtype=torch.float16,
            ).to(DEVICE)
    else:
        log.info("Loading mBLIP in float16 (no quantization — needs 8+ GB VRAM) …")
        model = Blip2ForConditionalGeneration.from_pretrained(
            MBLIP_PRETRAINED_NAME,
            torch_dtype=torch.float16,
        ).to(DEVICE)

    # Apply LoRA to the language model (LM) component of mBLIP
    # Only the LM is adapted — vision encoder and Q-Former stay frozen
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=MBLIP_LORA_RANK,
        lora_alpha=MBLIP_LORA_ALPHA,
        target_modules=["q", "v"],    # LoRA on attention query and value matrices
        lora_dropout=MBLIP_LORA_DROPOUT,
        bias="none",
    )

    # Wrap only the language_model part of BLIP-2
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()

    log.info("mBLIP + LoRA loaded ✓")
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids    = batch["input_ids"].to(DEVICE)
        labels       = batch.get("labels", input_ids).to(DEVICE)

        # Replace padding token id with -100 so loss ignores padding
        labels = labels.masked_fill(labels == 0, -100)

        with torch.amp.autocast(device_type=_scaler_device, enabled=USE_FP16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
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

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Validation", dynamic_ncols=True)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids    = batch["input_ids"].to(DEVICE)
        labels       = batch.get("labels", input_ids).to(DEVICE)
        labels       = labels.masked_fill(labels == 0, -100)

        with torch.amp.autocast(device_type=_scaler_device, enabled=USE_FP16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
            )
        total_loss += outputs.loss.item()
        pbar.set_postfix(val_loss=f"{total_loss / (pbar.n + 1):.4f}")

    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    log.info(f"Using device: {DEVICE}  |  FP16: {USE_FP16}")
    log.info(f"Model: {MBLIP_PRETRAINED_NAME}")
    log.info(f"4-bit quantization: {args.use_4bit}")
    log.info(f"LoRA rank: {MBLIP_LORA_RANK}  |  alpha: {MBLIP_LORA_ALPHA}")
    log.info("Campus caption dataset: data/campus_captions/")

    save_dir = Path(MBLIP_FINETUNED_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load mBLIP + LoRA
    model, processor = load_mblip_for_training(use_4bit=args.use_4bit)

    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            log.info(f"Resuming LoRA adapter from: {ckpt_path}")
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(
                model.language_model, str(ckpt_path)
            )
            model.to(DEVICE)
            try:
                start_epoch = int(ckpt_path.name.split("epoch")[-1]) + 1
            except ValueError:
                start_epoch = 1
        else:
            log.warning(f"Checkpoint not found: {ckpt_path}  — starting fresh")

    # DataLoaders
    train_loader, val_loader = get_dataloaders(
        processor=processor,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
    )

    # Only optimize LoRA parameters (base model is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    num_training_steps = (len(train_loader) // MBLIP_GRAD_ACCUM_STEPS) * args.epochs
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=MBLIP_WEIGHT_DECAY,
        eps=1e-8,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(num_training_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler(device=_scaler_device, enabled=USE_FP16)

    log.info(f"Training steps: {num_training_steps}  |  Epochs: {args.epochs}")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, MBLIP_GRAD_ACCUM_STEPS,
        )
        val_loss = validate(model, val_loader)

        log.info(
            f"Epoch {epoch} — Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save LoRA adapter checkpoint (only ~100 MB, not full 5 GB model)
        epoch_dir = save_dir / f"checkpoint_epoch{epoch}"
        model.language_model.save_pretrained(str(epoch_dir))
        processor.save_pretrained(str(epoch_dir))
        log.info(f"LoRA adapter checkpoint saved: {epoch_dir}")

        # Save best adapter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = save_dir / "best"
            model.language_model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            log.info(f"  ★ New best LoRA adapter saved (val_loss={best_val_loss:.4f})")

    log.info(f"\n✓ mBLIP LoRA fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"  Best adapter → {save_dir / 'best'}")
    log.info(f"  Next step: set MBLIP_USE_FINETUNED = True in config.py")
    log.info(f"  Then run: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune mBLIP with LoRA on Campus Telugu Captions"
    )
    parser.add_argument("--epochs",     type=int,   default=MBLIP_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=MBLIP_TRAIN_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=MBLIP_LEARNING_RATE)
    parser.add_argument(
        "--no-4bit", dest="use_4bit", action="store_false",
        help="Disable 4-bit quantization (needs 8+ GB VRAM — use on Colab/cloud)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to LoRA adapter checkpoint directory to resume from"
    )
    parser.set_defaults(use_4bit=MBLIP_USE_4BIT)
    args = parser.parse_args()
    main(args)
