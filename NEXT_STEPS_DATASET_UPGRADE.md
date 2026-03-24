# Next Steps — mBLIP Campus Navigation

> Updated March 2026 — mBLIP migration complete.

## Immediate Actions Required

### 1. Create Your Campus Caption Dataset ⬅ DO THIS FIRST
Follow **[DATASET_CREATION_GUIDE.md](DATASET_CREATION_GUIDE.md)** to:
- Take 300–500 photos on your college campus
- Write one Telugu caption per photo
- Format as `data/campus_captions/train.json` and `val.json`

### 2. Download YOLO Campus Detection Datasets
```bash
python data/download_datasets.py --dataset manual-info
```
Download the 4 Roboflow/Kaggle datasets listed.

### 3. Install New Dependencies
```bash
pip install -r requirements.txt
```
Key new packages: `peft` (LoRA training), `bitsandbytes` (4-bit quantization)

---

## Training Sequence (after dataset is ready)

```bash
# 1. Train YOLO (campus object detection)
python training/train_detector.py --dataset campus

# 2. Fine-tune mBLIP with LoRA (campus Telugu captions)
python training/train_captioner.py

# 3. Evaluate mBLIP
python training/evaluate.py

# 4. Run the app
python main.py
```

---

## Testing mBLIP Zero-Shot (Before Training)

You can test mBLIP's Telugu ability right now, before any fine-tuning:

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

proc  = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Gregor/mblip-mt0-xl", torch_dtype=torch.float16
).to("cuda")

img   = Image.open("your_campus_photo.jpg")
inp   = proc(images=img, text="Describe this campus scene in Telugu:", return_tensors="pt").to("cuda")
ids   = model.generate(**inp, max_new_tokens=80)
print(proc.decode(ids[0], skip_special_tokens=True))
```

---

## Cloud Training (Better Quality, Free)

If you want full float16 training (better quality, needs 12+ GB VRAM):

| Platform | VRAM | Cost | Command |
|---|---|---|---|
| Google Colab T4 | 15 GB | Free | `python training/train_captioner.py --no-4bit --batch-size 4` |
| Kaggle P100 | 16 GB | Free | Same |
| RunPod RTX 3090 | 24 GB | ~$0.4/hr | Same |
