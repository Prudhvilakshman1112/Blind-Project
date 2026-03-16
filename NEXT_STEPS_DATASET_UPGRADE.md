# NEXT STEP — Dataset Upgrade for College/Indoor Navigation
# Created: 2026-03-12
# Purpose: Train the model to detect objects in college premises for blind navigation

## Goal
Replace the generic COCO-only training with a combination of datasets that
cover indoor/campus environments — so the app can accurately help a blind
person navigate a university building or campus.

---

## Datasets to Download

### 1. Indoor Objects Detection (Blind-focused) — Kaggle
- URL: https://www.kaggle.com (search: "Indoor Objects Detection blind")
- Format: YOLO-ready
- Classes: door, openedDoor, cabinetDoor, window, chair, table, cabinet, sofa, pole
- Why: Built specifically for obstacle detection for blind people indoors

### 2. Door + Stairs + Chairs + Toilet Detection — Roboflow Universe
- URL: https://universe.roboflow.com (search: "door stairs chairs detection")
- Format: YOLO-ready (export as YOLOv11)
- Classes: door, stairs, chair, toilet
- Why: Exactly the objects a blind person needs in a building/campus

### 3. Indoor-objects by SmartCane — Roboflow Universe
- URL: https://universe.roboflow.com (search: "smartcane indoor objects")
- Format: YOLO-ready
- Classes: chair, table, door
- Why: Made by an assistive smartcane team — same use case

### 4. Door, Windows and Stairs Dataset — Kaggle
- URL: https://www.kaggle.com (search: "Door Windows Stairs Dataset Annotated")
- Format: Annotated images
- Classes: door, window, stairs
- Why: Stairs are the biggest fall risk for blind people

### 5. BLV Navigation Dataset (90 objects) — arXiv 2024
- URL: https://arxiv.org (search: "Dataset Crucial Object Recognition Blind Low-Vision Navigation 2024")
- Format: 21 videos, 90 object classes
- Classes: 90 objects chosen by blind people themselves in a research study
- Why: The most scientifically validated list for blind navigation

---

## Strategy — Combine All Datasets

  Final Training Data =
    MS-COCO (80 classes)             <- already in the project
  + Indoor Objects (Kaggle)          <- indoor obstacles
  + Door/Stairs/Chairs (Roboflow)    <- building navigation
  + BLV 90-object dataset (arXiv)    <- scientifically chosen
  + Own college photos (50-100 imgs) <- college-specific objects

Use Roboflow (free) to merge all datasets and export as a single
YOLOv11 format zip.

---

## What This Will Cover After Training

  Chairs, tables, sofas, cabinets  — indoor furniture
  Doors (open/closed), windows     — entry/exit navigation
  Stairs, steps, ramps             — fall prevention
  Poles, pillars                   — collision hazard
  Persons, bags, bicycles          — moving obstacles
  Toilet, sink                     — room identification
  Signs, exits                     — wayfinding
  90 BLV-validated objects         — scientifically proven critical list

---

## Training Command (after downloading datasets)

  # Use small model for better accuracy (recommended over nano)
  python training/train_detector.py --model yolo11s.pt

---

## Steps in Order

  1. Go to Roboflow Universe → download datasets listed above → export as YOLOv11
  2. Go to Kaggle → download datasets listed above
  3. Merge all in Roboflow (free) → export as one combined YOLO dataset
  4. Place merged dataset in data/ folder
  5. Run: python training/train_detector.py --model yolo11s.pt
  6. After training: set YOLO_USE_CUSTOM = True in config.py
  7. Run main application: python main.py
