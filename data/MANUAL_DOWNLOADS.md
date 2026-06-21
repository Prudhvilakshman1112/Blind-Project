# Manual Dataset Downloads — Campus Navigation (Roboflow / Kaggle)

These datasets **cannot be auto-downloaded** (require free account login).
Follow the steps below, then place each dataset in the correct folder.

> **Note:** The project no longer uses full MS-COCO, VizWiz, or AI4Bharat IndicCOCO.
> All YOLO training now uses campus-specific Roboflow datasets only (fast + accurate).
> For Telugu captions (mBLIP fine-tuning), create your own campus dataset:
> `python data/download_datasets.py --dataset campus-setup`
> Then follow **DATASET_CREATION_GUIDE.md** in the project root.

---

## Dataset 1 — SCIN Indoor Navigation System (Roboflow)

**What it is:** 544 annotated indoor navigation images  
**Classes:** `door`, `stairs`  
**Direct URL:** https://universe.roboflow.com/scin/indoor-navigation-system

### Steps:
1. Open: https://universe.roboflow.com/scin/indoor-navigation-system
2. Click **Download Dataset**
3. Select format: **YOLOv11** → click **Download** (zip)
4. Extract the zip
5. Place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\scin_indoor\
   ```
   Expected structure:
   ```
   scin_indoor/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/
   ```

---

## Dataset 2 — Akhash Indoor Navigation (Roboflow)

**What it is:** 1,115 annotated indoor navigation images  
**Classes:** `door`, `person`, `elevator`, `stair sign`  
**Direct URL:** https://universe.roboflow.com/akhash/indoor-navigation

### Steps:
1. Open: https://universe.roboflow.com/akhash/indoor-navigation
2. Click **Download Dataset**
3. Select format: **YOLOv11** → click **Download** (zip)
4. Extract and place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\akhash_indoor\
   ```

---

## Dataset 3 — Indoor Navigation For The Blind (Roboflow)

**What it is:** Indoor assistive navigation dataset  
**Classes:** `door`, `stairs`, `pole`, `chair`, `table`  
**Search:** https://universe.roboflow.com — search **"IndoorNavigationForTheBlinds"**

### Steps:
1. Go to https://universe.roboflow.com
2. Search: **IndoorNavigationForTheBlinds**
3. Open the dataset → click **Download Dataset**
4. Select format: **YOLOv11** → click Download (zip)
5. Extract and place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\blind_indoor\
   ```

---

## Dataset 4 — Stairs Detection Dataset (Kaggle)

**What it is:** 1,000 JPEG stair images with YOLO-format bounding boxes  
**Classes:** `stairs`  
**Search:** https://www.kaggle.com — search **"Stairs Detection YOLO Samuel Ayman"**

### Steps:
1. Go to https://www.kaggle.com
2. Sign in (or create a free account)
3. Search: **"Stairs Detection YOLO Samuel Ayman"**
4. Click **Download** (zip)
5. Extract and place contents inside:
   ```
   d:\Blind-Project\data\indoor_campus\stairs_kaggle\
   ```

---

## After Downloading All Datasets

Run the YOLO campus training:
```bash
python training/train_detector.py --dataset campus
```

The training script auto-discovers all sub-folders inside `data/indoor_campus/`
and trains on the combined campus-specific detection classes.

---

## Folder Summary

```
data/
├── campus_captions/     ← Your own campus photos + Telugu captions
│   ├── train.json           (replace sample entries with your data)
│   ├── val.json             (replace sample entries with your data)
│   └── images/              (place your .jpg campus photos here)
└── indoor_campus/       ← MANUAL DOWNLOAD (this guide)
    ├── scin_indoor/     ← Roboflow SCIN  (door, stairs)
    ├── akhash_indoor/   ← Roboflow Akhash (door, person, elevator, stair sign)
    ├── blind_indoor/    ← Roboflow Blind (door, stairs, pole, chair, table)
    └── stairs_kaggle/   ← Kaggle stairs (stairs)
```

## Verify Downloads

Run at any time to see what is ready:
```bash
python data/download_datasets.py --dataset verify
```
