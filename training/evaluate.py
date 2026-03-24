"""
training/evaluate.py
─────────────────────
Evaluates the mBLIP model (base or LoRA fine-tuned) using BLEU (1–4) and
METEOR metrics on your campus caption validation set.

mBLIP UPGRADE (March 2026):
  Uses Blip2Processor and Blip2ForConditionalGeneration.
  Evaluates against your human-collected campus caption dataset.
  Passes MBLIP_PROMPT to the model so it generates in Telugu.

Saves a detailed JSON report to logs/eval_report.json.

Usage:
    python training/evaluate.py
    python training/evaluate.py --model checkpoints/mblip_campus/best
    python training/evaluate.py --max-samples 100
    python training/evaluate.py --no-4bit   (for cloud GPUs)
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, USE_FP16,
    MBLIP_PRETRAINED_NAME, MBLIP_FINETUNED_PATH,
    MBLIP_USE_4BIT, MBLIP_PROMPT,
    MBLIP_MAX_NEW_TOKENS, MBLIP_NUM_BEAMS,
    CAMPUS_CAPTION_DIR, LOGS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ensure NLTK data is present
for pkg in ["wordnet", "averaged_perceptron_tagger", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_eval(model_path: str, use_4bit: bool):
    """
    Loads mBLIP for evaluation.
    If model_path points to a LoRA adapter directory, loads the LoRA adapter
    on top of the base mBLIP model.
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained(MBLIP_PRETRAINED_NAME)

    # Check if model_path is a LoRA adapter or the base model
    peft_config_file = Path(model_path) / "adapter_config.json"
    is_lora_adapter  = peft_config_file.exists()

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                MBLIP_PRETRAINED_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ImportError:
            log.warning("4-bit unavailable. Loading in float16.")
            model = Blip2ForConditionalGeneration.from_pretrained(
                MBLIP_PRETRAINED_NAME, torch_dtype=torch.float16
            ).to(DEVICE)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            MBLIP_PRETRAINED_NAME, torch_dtype=torch.float16
        ).to(DEVICE)

    if is_lora_adapter:
        try:
            from peft import PeftModel
            model.language_model = PeftModel.from_pretrained(
                model.language_model, model_path
            )
            log.info(f"LoRA adapter loaded from: {model_path} ✓")
        except ImportError:
            log.warning("PEFT not installed — evaluating base mBLIP without LoRA adapter.")

    model.eval()
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Caption Generator
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(model, processor, image: Image.Image) -> str:
    """Generates a single Telugu caption for an image using mBLIP."""
    inputs = processor(
        images=image,
        text=MBLIP_PROMPT,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.amp.autocast(
        device_type="cuda" if DEVICE == "cuda" else "cpu", enabled=USE_FP16
    ):
        ids = model.generate(
            **inputs,
            max_new_tokens=MBLIP_MAX_NEW_TOKENS,
            num_beams=MBLIP_NUM_BEAMS,
            early_stopping=True,
        )
    return processor.decode(ids[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    log.info(f"Loading model for evaluation: {args.model}")
    model, processor = load_model_for_eval(args.model, use_4bit=args.use_4bit)

    # Load campus caption val annotations
    val_json = CAMPUS_CAPTION_DIR / "val.json"
    img_dir  = CAMPUS_CAPTION_DIR / "images"

    if not val_json.exists():
        log.error(
            f"Campus val annotations not found: {val_json}\n"
            "Setup first: python data/download_datasets.py --dataset campus-setup\n"
            "Then add your campus images and Telugu captions per DATASET_CREATION_GUIDE.md"
        )
        sys.exit(1)

    with open(val_json, encoding="utf-8") as f:
        records = json.load(f)

    if len(records) <= 1:
        log.warning(
            "Only sample data found in val.json.\n"
            "Please add your own campus images per DATASET_CREATION_GUIDE.md before evaluating."
        )

    if args.max_samples:
        records = records[:args.max_samples]

    # Generate predictions
    references, hypotheses = [], []
    results = []
    smoother = SmoothingFunction().method1

    for rec in tqdm(records, desc="Evaluating"):
        fname   = rec.get("file_name", "")
        caption = rec.get("caption", "").strip()
        if not caption or not fname:
            continue

        img_path = img_dir / fname
        if not img_path.exists():
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            image = Image.open(img_path).convert("RGB")

        predicted = generate_caption(model, processor, image)

        hypotheses.append(predicted.lower().split())
        references.append([caption.lower().split()])

        results.append({
            "file_name": fname,
            "predicted": predicted,
            "reference": caption,
        })

    if not results:
        log.error("No valid samples to evaluate.")
        sys.exit(1)

    # Compute corpus BLEU
    bleu1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0), smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5,0,0), smoothing_function=smoother)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3,1/3,1/3,0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother)

    meteor_scores = [
        meteor_score(refs, hyp)
        for refs, hyp in zip(references, hypotheses)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    report = {
        "model":         args.model,
        "base_model":    MBLIP_PRETRAINED_NAME,
        "dataset":       "Human campus caption dataset (data/campus_captions/val.json)",
        "num_samples":   len(results),
        "bleu_1":        round(bleu1, 4),
        "bleu_2":        round(bleu2, 4),
        "bleu_3":        round(bleu3, 4),
        "bleu_4":        round(bleu4, 4),
        "meteor":        round(avg_meteor, 4),
        "samples":       results[:20],
    }

    report_path = LOGS_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 55)
    log.info("EVALUATION RESULTS  (mBLIP — Telugu Campus Captions)")
    log.info("=" * 55)
    log.info(f"  Samples  : {report['num_samples']}")
    log.info(f"  BLEU-1   : {report['bleu_1']:.4f}")
    log.info(f"  BLEU-2   : {report['bleu_2']:.4f}")
    log.info(f"  BLEU-3   : {report['bleu_3']:.4f}")
    log.info(f"  BLEU-4   : {report['bleu_4']:.4f}  (target ≥ 0.25)")
    log.info(f"  METEOR   : {report['meteor']:.4f}  (target ≥ 0.28)")
    log.info(f"  Report   : {report_path}")

    if report["bleu_4"] >= 0.25 and report["meteor"] >= 0.28:
        log.info("  ✓ Metrics meet target — model ready for deployment!")
    else:
        log.info("  ✗ Metrics below target — add more campus images or train more epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate mBLIP Telugu captioner on campus dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(MBLIP_FINETUNED_PATH) / "best"),
        help="Path to LoRA adapter dir (or base model name for zero-shot eval)"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--no-4bit", dest="use_4bit", action="store_false",
        help="Disable 4-bit quantization (needs 8+ GB VRAM)"
    )
    parser.set_defaults(use_4bit=MBLIP_USE_4BIT)
    args = parser.parse_args()
    evaluate(args)
