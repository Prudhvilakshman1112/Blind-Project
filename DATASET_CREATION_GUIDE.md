# Dataset Creation Guide — Campus Telugu Caption Dataset for mBLIP

> **This is the most important step for making this project work.** mBLIP already knows Telugu — you just need to teach it your specific campus environment.
>
> Updated: June 2026 — Reflects actual project structure, JSON format, and training pipeline.

---

## Why You Need This Dataset

mBLIP (`Gregor/mblip-mt0-xl`) is a multilingual BLIP-2 model that already speaks Telugu natively (one of 96 languages). However, it has never seen your college campus. By providing 300–500 photos of your campus with Telugu captions, you teach it to describe:

- "మెట్లు చాలా దగ్గరగా ఉన్నాయి" (Stairs are very close)
- "నడవలో ఒక వ్యక్తి నడుస్తున్నాడు" (A person is walking in the corridor)
- "తలుపు తెరిచి ఉంది" (The door is open)

**You do NOT need thousands of images.** Even 300 good campus photos with native Telugu captions will dramatically improve quality.

> **Important:** All previously referenced HuggingFace Telugu image-caption datasets have been confirmed deleted or private as of March 2026 (`Hardik15/telugu-image-captions`, `FutureBeeAI/telugu-image-captions`, `ai4bharat/IndicCOCO`). Your own campus dataset is the ONLY reliable path forward.

---

## Step 1 — Set Up the Folder Structure

```bash
python data/download_datasets.py --dataset campus-setup
```

This creates:
```
data/campus_captions/
├── train.json    ← 2 sample entries (you will replace these)
├── val.json      ← 1 sample entry  (you will replace these)
└── images/       ← place your .jpg campus photos here
```

The sample JSON files contain placeholder entries in the exact format the training pipeline expects. **Replace them** with your own data.

---

## Step 2 — Take Photos

### How Many
- **Minimum:** 300 photos (LoRA fine-tuning works well with this)
- **Target:** 500 photos (better campus-specific accuracy)
- **Recommended:** 500–800 photos if you can collect them

### What Scenarios to Cover

| Category | Examples | # Photos |
|---|---|---|
| **Stairs & Steps** | Ascending, descending, side view, close-up, ramp | 40–60 |
| **Doors & Entrances** | Open door, closed door, glass door, double door | 40–60 |
| **Corridors & Hallways** | Empty, with people, with benches, narrow passages | 40–60 |
| **People** | Person walking, standing, carrying backpack | 40–60 |
| **Furniture** | Bench, chair, table, in different arrangements | 30–40 |
| **Poles & Hazards** | Lamp poles, bollards, sign poles, pillars | 20–30 |
| **Outdoor Paths** | Campus pathways, gardens, ramps | 30–40 |
| **Classrooms** | Desks, chairs, boards, projectors, laptops | 20–30 |
| **Canteen / Open Areas** | Tables, crowded spaces, queues | 20–30 |
| **Windows** | Glass walls, corridor windows | 10–15 |
| **Signs & Boards** | Notice boards, room number signs (OCR training) | 20–30 |

### Photo Tips
- Take photos from **eye level** (as if the blind user is standing)
- Include photos with **objects at different distances** (close, medium, far)
- Take photos in **different lighting** (morning, afternoon, indoor, outdoor)
- **Avoid blurry photos** — clear images train better
- Use your phone camera — **no special equipment needed**
- Try both landscape and portrait orientations

---

## Step 3 — Write Telugu Captions

### Caption Guidelines

Each photo needs **one clear Telugu sentence** describing what a blind person should know for safe navigation.

✅ **Good captions** (specific, useful for navigation):
```
మెట్లు ముందు ఉన్నాయి, జాగ్రత్తగా దిగండి.
(Stairs are ahead, please descend carefully.)

నడవలో ఒక వ్యక్తి నడుస్తున్నాడు, కుడి వైపు బెంచీ ఉంది.
(A person is walking in the corridor, a bench is on the right side.)

తలుపు తెరిచి ఉంది, లోపలికి ప్రవేశించవచ్చు.
(The door is open, you can enter inside.)

ముందు స్తంభం ఉంది, ఎడమ వైపు నడవండి.
(There is a pole ahead, walk to the left side.)
```

❌ **Avoid vague captions**:
```
"ఇక్కడ ఒక ఫోటో ఉంది"  (Here is a photo)
"చాలా బాగుంది"         (Looks very beautiful)
"కాలేజ్ ఉంది"          (There is a college)
```

### Who Should Write Captions
- **Ideal:** Native Telugu speaker who understands the campus
- **Acceptable:** You (with basic Telugu knowledge) + review by Telugu speaker
- **Tool:** Google Input Tools (https://www.google.com/inputtools/) to type Telugu on PC

---

## Step 4 — Create the JSON Files

### JSON Format (train.json and val.json)

The training pipeline (`data/dataset_loader.py → CampusCaptionDataset`) reads this exact format:

```json
[
  {
    "idx": 0,
    "file_name": "train_000000.jpg",
    "caption": "నడవలో ఒక వ్యక్తి నడుస్తున్నాడు, ముందు మెట్లు ఉన్నాయి."
  },
  {
    "idx": 1,
    "file_name": "train_000001.jpg",
    "caption": "తలుపు తెరిచి ఉంది, ప్రవేశ ద్వారం కళాశాల ముందు భవనంలో ఉంది."
  }
]
```

**Fields:**
- `idx` — Sequential integer index (0, 1, 2, …)
- `file_name` — Exact filename of the image in `data/campus_captions/images/`
- `caption` — Telugu caption string (UTF-8)

### Naming Your Images

Save your images with sequential names:
- Training images: `train_000000.jpg`, `train_000001.jpg`, `train_000002.jpg`, …
- Validation images: `val_000000.jpg`, `val_000001.jpg`, …

Place **all images** in: `data/campus_captions/images/`

### Train / Validation Split
- Use **80% for training** (train.json) — e.g., 400 photos
- Use **20% for validation** (val.json) — e.g., 100 photos
- Validation images must be **different from training images**

---

## Step 5 — Using Label Studio (Optional, Recommended for Larger Datasets)

[Label Studio](https://labelstud.io) is a free annotation tool that makes it easy to view images and type captions.

```bash
pip install label-studio
label-studio
```

1. Create new project → Select **"Image Captioning"** task
2. Upload your campus images
3. Write Telugu captions for each image
4. Export as JSON → reformat to match the schema above

---

## Step 6 — Verify Your Dataset

```bash
python data/download_datasets.py --dataset verify
```

You should see:
```
✓ Campus captions : 400 train + 100 val pairs
```

If you see "only sample data (2 train entries)", you still need to add your own images and replace the sample JSON files.

---

## Step 7 — Ready to Train!

Once your dataset is ready:
```bash
python training/train_captioner.py
```

See **TRAINING_GUIDE.md** for full training instructions and VRAM requirements.

---

## Quick Start (Minimum Viable Dataset)

If you want to test the pipeline first with just a few images:

1. Take **10 photos** on campus
2. Write 10 Telugu captions
3. Place 8 in `train.json` (idx 0–7), 2 in `val.json` (idx 0–1)
4. Place images in `data/campus_captions/images/`
5. Run 1 epoch of training to verify the pipeline works end-to-end

```bash
python training/train_captioner.py --epochs 1
```

This is enough to test the full pipeline. Add more photos gradually for better results.

---

## How the Dataset is Used in Training

The `data/dataset_loader.py` `CampusCaptionDataset` class:
1. Reads `train.json` / `val.json`
2. Loads each image from `images/` subdirectory (falls back to grey placeholder if missing)
3. Applies image augmentations (random crop, brightness, blur, noise) — see `data/augmentations.py`
4. Tokenizes the Telugu caption using `Blip2Processor` (mBLIP tokenizer)
5. Returns `pixel_values`, `input_ids`, and `labels` for LoRA training

The mBLIP model is prompted with `MBLIP_PROMPT = "క్లుప్తంగా వివరించు:"` (Describe briefly) and trained to complete the prompt with your Telugu captions.

---

## Training Prompt

The model receives this prompt during training and inference:
```
"క్లుప్తంగా వివరించు:"
```

This is set in `config.py → MBLIP_PROMPT`. It tells mBLIP to respond in Telugu. Your captions teach it campus-specific vocabulary for that response.

---

## Telugu Vocabulary Reference (for Caption Writers)

| English | Telugu | Pronunciation |
|---|---|---|
| Stairs / Steps | మెట్లు | Metlu |
| Door | తలుపు | Talapu |
| Open door | తెరిచిన తలుపు | Terichina talapu |
| Person / People | వ్యక్తి / జనులు | Vyakti / Janulu |
| Corridor / Hallway | నడవ | Nadava |
| Bench | బెంచీ | Benchi |
| Chair | కుర్చీ | Kurchi |
| Table | టేబుల్ | Tebul |
| Ahead / Front | ముందు | Mundu |
| Left | ఎడమ | Edama |
| Right | కుడి | Kudi |
| Very close | చాలా దగ్గరగా | Chala daggaragaa |
| Nearby | దగ్గరలో | Daggaralō |
| Warning / Careful | జాగ్రత్త | Jaagratta |
| Pole / Pillar | స్తంభం | Stambham |
| Ramp | రాంప్ | Ramp |
| Backpack | బ్యాగు | Byagu |
| Window | కిటికీ | Kitiki |
| Bicycle | సైకిల్ | Saikil |
| Motorcycle | మోటర్‌సైకిల్ | Mōṭar‌saikil |
| Car | కారు | Kaaru |
| Laptop | ల్యాప్‌టాప్ | Lyāpṭāp |
| Campus | కాలేజ్ ఆవరణ | Kālēj āvaraṇa |
| Building | భవనం | Bhawanam |
| Sign / Board | బోర్డు | Bordu |
| Notice board | నోటీసు బోర్డు | Nōṭīsu bordu |

---

## After Training

Once training is complete and you've verified metrics (see `TRAINING_GUIDE.md`):

```python
# In config.py — set this flag to use your fine-tuned LoRA adapter:
MBLIP_USE_FINETUNED = True
```

Then run the application:
```bash
python main.py
```

The app will use your campus-tuned mBLIP adapter loaded from `checkpoints/mblip_campus/best/`.
