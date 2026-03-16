# Manual Dataset Downloads — Indoor Campus Navigation

These datasets **cannot be auto-downloaded** (require account login).
Follow the steps below, then place each dataset in the correct folder.

---

## Dataset 1 — Indoor Objects Detection (Kaggle)

**What it is:** YOLO-format dataset for blind/assistive indoor navigation.
**Classes:** door, openedDoor, cabinetDoor, window, chair, table, cabinet, sofa, pole

### Steps:
1. Go to https://www.kaggle.com
2. Sign in (or create a free account)
3. Search: **"Indoor Objects Detection blind"**
4. Click **Download** (zip file)
5. Extract the zip
6. Place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\indoor_objects\
   ```
   Expected structure:
   ```
   indoor_objects/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

---

## Dataset 2 — Door + Stairs + Chairs + Toilet (Roboflow)

**What it is:** YOLO object detection dataset for building navigation.
**Classes:** door, stairs, chair, toilet

### Steps:
1. Go to https://universe.roboflow.com
2. Search: **"door stairs chairs detection"**
3. Open the dataset → click **Download Dataset**
4. Select format: **YOLOv11** → click Download (zip)
5. Extract and place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\door_stairs\
   ```

---

## Dataset 3 — SmartCane Indoor Objects (Roboflow)

**What it is:** YOLO dataset built by a smartcane assistive device team.
**Classes:** chair, table, door

### Steps:
1. Go to https://universe.roboflow.com
2. Search: **"smartcane indoor objects"**
3. Open the dataset → click **Download Dataset**
4. Select format: **YOLOv11** → click Download (zip)
5. Extract and place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\smartcane\
   ```

---

## Dataset 4 — Stairs and Doors (Kaggle)

**What it is:** Annotated images of stairs, doors, windows.
**Classes:** door, window, stairs

### Steps:
1. Go to https://www.kaggle.com
2. Search: **"Door Windows Stairs Dataset Annotated"**
3. Download and extract
4. Convert to YOLO format if needed (Roboflow free converter helps)
5. Place in:
   ```
   d:\Blind-Project\data\indoor_campus\stairs_doors\
   ```

---

## After Downloading All Datasets

Run the combined YOLO training:
```bash
python training/train_detector.py --dataset combined --model yolo11s.pt
```

The training script will auto-detect all sub-folders inside `data/indoor_campus/`
and merge them with COCO into one training run.

---

## Folder Summary

```
data/
├── vizwiz/              ← Auto-downloaded (python data/download_datasets.py --dataset vizwiz)
├── coco/                ← Auto-downloaded (python data/download_datasets.py --dataset coco)
├── indic_caption/       ← Auto-downloaded (python data/download_datasets.py --dataset indic)
└── indoor_campus/       ← MANUAL DOWNLOAD (this guide)
    ├── indoor_objects/  ← Kaggle
    ├── door_stairs/     ← Roboflow
    ├── smartcane/       ← Roboflow
    └── stairs_doors/    ← Kaggle
```
