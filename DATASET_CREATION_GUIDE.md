# Dataset Creation Guide — Campus Telugu Caption Dataset for mBLIP

> **This is the most important step for making this project work.** mBLIP already knows Telugu — you just need to teach it your specific campus environment.

---

## Why You Need This Dataset

mBLIP (`Gregor/mblip-mt0-xl`) is a multilingual AI model that already speaks Telugu. However, it has never seen your college campus. By providing 300–500 photos of your campus with Telugu captions, you teach it to describe:

- "మెట్లు చాలా దగ్గరగా ఉన్నాయి" (Stairs are very close)
- "నడవలో ఒక వ్యక్తి నడుస్తున్నాడు" (A person is walking in the corridor)
- "తలుపు తెరిచి ఉంది" (The door is open)

**You do NOT need thousands of images.** Even 300 good campus photos with native Telugu captions will dramatically improve quality.

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

---

## Step 2 — Take Photos

### How Many
- **Minimum:** 300 photos (fine-tuning works with this)
- **Target:** 500 photos (better accuracy)
- **More is better**, but quality matters more than quantity

### What Scenarios to Cover

| Category | Examples | # Photos |
|---|---|---|
| **Stairs & Steps** | Ascending, descending, side view, close-up | 40–60 |
| **Doors & Entrances** | Open door, closed door, glass door, double door | 40–60 |
| **Corridors & Hallways** | Empty, with people, with benches | 40–60 |
| **People** | Person walking, standing, with backpack | 40–60 |
| **Furniture** | Bench, chair, table, in different arrangements | 30–40 |
| **Poles & Hazards** | Lamp poles, bollards, sign poles | 20–30 |
| **Outdoor Paths** | Campus pathways, gardens, ramps | 30–40 |
| **Classrooms** | Desks, chairs, boards, projectors | 20–30 |
| **Canteen / Open Areas** | Tables, crowded spaces, queues | 20–30 |

### Photo Tips
- Take photos from **eye level** (as if the blind user is standing)
- Include photos with **objects at different distances** (close, medium, far)
- Take photos in **different lighting** (morning, afternoon, indoor)
- **Avoid blurry photos** — clear images train better
- Use your phone camera — **no special equipment needed**

---

## Step 3 — Write Telugu Captions

### Caption Guidelines

Each photo needs **one clear Telugu sentence** describing what a blind person should know.

✅ **Good captions** (specific, useful for navigation):
```
మెట్లు ముందు ఉన్నాయి, జాగ్రత్తగా దిగండి.
(Stairs are ahead, please descend carefully.)

నడవలో ఒక వ్యక్తి నడుస్తున్నాడు, కుడి వైపు బెంచీ ఉంది.
(A person is walking in the corridor, a bench is on the right side.)

తలుపు తెరిచి ఉంది, లోపలికి ప్రవేశించవచ్చు.
(The door is open, you can enter inside.)
```

❌ **Avoid vague captions**:
```
"ఇక్కడ ఒక ఫోటో ఉంది"  (Here is a photo)
"చాలా బాగుంది"         (Looks very beautiful)
```

### Who Should Write Captions
- **Ideal:** Native Telugu speaker who understands the campus
- **Acceptable:** You (with basic Telugu knowledge) + review by Telugu speaker
- **Tool:** You can use Google Input Tools to type Telugu on PC

---

## Step 4 — Create the JSON Files

### JSON Format (train.json and val.json)

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

1. Create new project → Select "Image Captioning" task
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

---

## Step 7 — Ready to Train!

Once your dataset is ready:
```bash
python training/train_captioner.py
```

See **TRAINING_GUIDE.md** for full training instructions.

---

## Quick Start (Minimum Viable Dataset)

If you want to test the pipeline first with just a few images:

1. Take **10 photos** on campus
2. Write 10 Telugu captions
3. Place 8 in `train.json` (idx 0–7), 2 in `val.json` (idx 0–1)
4. Place images in `data/campus_captions/images/`
5. Run 1 epoch of training to verify everything works

This is enough to test the full pipeline. Add more photos gradually for better results.

---

## Telugu Vocabulary Reference (for Caption Writers)

| English | Telugu | Pronunciation |
|---|---|---|
| Stairs / Steps | మెట్లు | Metlu |
| Door | తలుపు | Talapu |
| Open door | తెరిచిన తలుపు | Terichina talapu |
| Person / People | వ్యక్తి / జనులు | Vyakti / Janulu |
| Corridor / Hallway | నడవ | Nadata |
| Bench | బెంచీ | Benchi |
| Chair | కుర్చీ | Kurchi |
| Table | టేబుల్ | Tebul |
| Ahead / Front | ముందు | Mundu |
| Left | ఎడమ | Edama |
| Right | కుడి | Kudi |
| Very close | చాలా దగ్గరగా | Chala daggaragaa |
| Nearby | దగ్గరలో | Daggaralō |
| Warning / Careful | జాగ్రత్త | Jaagratta |
| Pole | స్తంభం | Stambham |
| Ramp | రాంప్ | Ramp |
| Backpack | బ్యాగు | Byagu |
| Window | కిటికీ | Kitiki |
