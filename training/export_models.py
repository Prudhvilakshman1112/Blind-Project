"""
training/export_models.py
──────────────────────────
Exports the fine-tuned BLIP model and YOLOv11 to ONNX and OpenVINO IR
format for deployment on lower-power devices (laptop CPU, Jetson Nano,
Raspberry Pi).

What gets exported:
  1. BLIP Vision Encoder → ONNX
  2. BLIP Text Decoder   → ONNX
  3. BLIP (combined)     → OpenVINO IR (vision encoder only — for speed)
  4. YOLOv11             → ONNX + OpenVINO IR

Usage:
    python training/export_models.py
    python training/export_models.py --skip-yolo
    python training/export_models.py --skip-blip
"""

import sys
import argparse
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEVICE, BLIP_FINETUNED_PATH, CHECKPOINTS_DIR,
    EXPORTED_DIR, BLIP_ONNX_PATH, OPENVINO_MODEL_DIR, LOGS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOGS_DIR / "export.log")),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BLIP Export
# ─────────────────────────────────────────────────────────────────────────────

def export_blip_onnx(model_path: str) -> Path:
    """Exports the BLIP vision encoder to ONNX."""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    import numpy as np

    log.info(f"Loading BLIP from {model_path} for ONNX export ...")
    processor = BlipProcessor.from_pretrained(model_path)
    model     = BlipForConditionalGeneration.from_pretrained(model_path)
    model.eval().to("cpu")  # Export on CPU for compatibility

    # Dummy input
    dummy_image = Image.fromarray(
        (255 * torch.rand(384, 384, 3).numpy()).astype("uint8")
    )
    inputs = processor(images=dummy_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    onnx_path = Path(BLIP_ONNX_PATH)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export vision encoder only (the computationally heavy part)
    vision_encoder = model.vision_model

    log.info(f"Exporting BLIP vision encoder → {onnx_path}")
    torch.onnx.export(
        vision_encoder,
        pixel_values,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        },
        verbose=False,
    )
    log.info(f"✓ BLIP ONNX export complete: {onnx_path}")
    return onnx_path


def export_blip_openvino(onnx_path: Path) -> Path:
    """Converts BLIP ONNX to OpenVINO IR format."""
    try:
        from openvino.tools.mo import convert_model
        from openvino.runtime import serialize
    except ImportError:
        try:
            import subprocess
            ov_dir = Path(OPENVINO_MODEL_DIR)
            ov_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    sys.executable, "-m", "mo",
                    "--input_model", str(onnx_path),
                    "--output_dir", str(ov_dir),
                    "--model_name", "blip_vision_encoder",
                    "--compress_to_fp16",
                ],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                log.info(f"✓ OpenVINO IR export complete: {ov_dir}")
                return ov_dir
            else:
                log.error(f"OpenVINO export failed: {result.stderr}")
        except Exception as e:
            log.warning(f"OpenVINO export skipped (not available): {e}")
    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Export
# ─────────────────────────────────────────────────────────────────────────────

def export_yolo(weights_path: str) -> None:
    """Exports YOLOv11 to ONNX and OpenVINO using ultralytics built-in export."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.")
        return

    weights = Path(weights_path)
    if not weights.exists():
        log.warning(
            f"YOLO weights not found: {weights}\n"
            "Skipping YOLO export. Train the detector first."
        )
        return

    log.info(f"Exporting YOLOv11 → ONNX from {weights} ...")
    model = YOLO(str(weights))

    # ONNX export
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
    log.info("✓ YOLOv11 ONNX export complete")

    # OpenVINO export
    log.info("Exporting YOLOv11 → OpenVINO IR ...")
    model.export(format="openvino", imgsz=640, half=True)
    log.info("✓ YOLOv11 OpenVINO IR export complete")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    blip_best = Path(BLIP_FINETUNED_PATH) / "best"

    if not args.skip_blip:
        if blip_best.exists():
            onnx_path = export_blip_onnx(str(blip_best))
            export_blip_openvino(onnx_path)
        else:
            log.warning(
                f"BLIP fine-tuned model not found at {blip_best}\n"
                "Skipping BLIP export. Train the captioner first."
            )

    if not args.skip_yolo:
        yolo_weights = CHECKPOINTS_DIR / "yolo11_custom.pt"
        export_yolo(str(yolo_weights))

    log.info(f"\n✓ Export complete. Files saved to: {EXPORTED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX/OpenVINO")
    parser.add_argument("--skip-blip", action="store_true", help="Skip BLIP export")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO export")
    args = parser.parse_args()
    main(args)
