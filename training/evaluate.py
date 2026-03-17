"""
training/evaluate.py
─────────────────────
Evaluates the fine-tuned BLIP model using BLEU (1–4) and METEOR metrics
on the Telugu validation set.

WHAT CHANGED (2025 overhaul):
  Evaluation now runs on the Telugu caption validation split, NOT VizWiz.
  Metrics measure how well BLIP generates Telugu descriptions.

Saves a detailed JSON report to logs/eval_report.json.

Usage:
    python training/evaluate.py
    python training/evaluate.py --model checkpoints/blip_telugu/best
    python training/evaluate.py --max-samples 500
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, USE_FP16, BLIP_PRETRAINED_NAME, BLIP_FINETUNED_PATH,
    BLIP_MAX_NEW_TOKENS, BLIP_NUM_BEAMS, CAMPUS_CAPTION_DIR, LOGS_DIR,
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
# Caption Generator
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_caption(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
) -> str:
    """Generates a single caption for an image using beam search."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.amp.autocast(
        device_type="cuda" if DEVICE == "cuda" else "cpu", enabled=USE_FP16
    ):
        ids = model.generate(
            **inputs,
            max_new_tokens=BLIP_MAX_NEW_TOKENS,
            num_beams=BLIP_NUM_BEAMS,
            early_stopping=True,
        )
    return processor.decode(ids[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    log.info(f"Loading model from: {args.model}")
    processor = BlipProcessor.from_pretrained(args.model)
    model     = BlipForConditionalGeneration.from_pretrained(args.model)
    model.eval().to(DEVICE)

    # Load Telugu val annotations
    val_json = CAMPUS_CAPTION_DIR / "val.json"
    img_dir  = CAMPUS_CAPTION_DIR / "images"

    if not val_json.exists():
        log.error(
            f"Telugu val annotations not found: {val_json}\n"
            "Download first: python data/download_datasets.py --dataset telugu"
        )
        sys.exit(1)

    with open(val_json, encoding="utf-8") as f:
        records = json.load(f)

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
            # Use blank image if missing
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

    # METEOR (sample average)
    meteor_scores = [
        meteor_score(refs, hyp)
        for refs, hyp in zip(references, hypotheses)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    report = {
        "model":       args.model,
        "dataset":     "Hardik15/telugu-image-captions (val split)",
        "num_samples": len(results),
        "bleu_1":      round(bleu1, 4),
        "bleu_2":      round(bleu2, 4),
        "bleu_3":      round(bleu3, 4),
        "bleu_4":      round(bleu4, 4),
        "meteor":      round(avg_meteor, 4),
        "samples":     results[:20],
    }

    report_path = LOGS_DIR / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 50)
    log.info("EVALUATION RESULTS  (Telugu Captions)")
    log.info("=" * 50)
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
        log.info("  ✗ Metrics below target — consider more training epochs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Telugu BLIP captioner"
    )
    parser.add_argument("--model",       type=str, default=f"{BLIP_FINETUNED_PATH}/best")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(args)
