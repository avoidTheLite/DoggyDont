# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DoggyDont detects when a dog is squatting (as if going to the bathroom) in video footage from overhead or steep downward-facing cameras. It uses YOLOv8 for dog detection and orientation-invariant bounding-box compactness for posture classification.

## Setup

**WSL/Ubuntu:**
```bash
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
python3 -m venv venv
source venv/bin/activate
pip install -r dog_posture_detection/requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r dog_posture_detection\requirements.txt
```

## Common Commands

All commands run from `dog_posture_detection/`:

```bash
# Run detection
python detect.py path/to/video.mp4
python detect.py path/to/video.mp4 --threshold 0.70   # override squat threshold
python detect.py path/to/video.mp4 --no-video          # skip annotated video output

# Play annotated output
python viewer.py                          # plays output/annotated_output.avi
python viewer.py path/to/video.avi        # plays specific file
# Controls: SPACE=pause, D=step forward, A=step back, Q/ESC=quit

# Evaluate against ground truth
python evaluate.py --ground-truth labels.csv
python evaluate.py --ground-truth labels.csv --detections output/detections.csv
```

## Architecture

### Core Classification (`posture.py`)

The posture classifier is orientation-invariant — it uses the **compactness ratio** (`minor_axis / major_axis`) of the bounding box, where major/minor are simply the longer/shorter dimensions. This works for dogs facing any direction:
- **Standing**: elongated box → low compactness (~0.25–0.50)
- **Squatting**: legs splay out, box becomes square → high compactness (~0.65–0.90)
- Threshold is configurable via `config.yaml` (`squat_threshold: 0.75`) or CLI `--threshold`

### Detection Pipeline (`detect.py`)

1. Loads YOLOv8n (auto-downloads ~6MB on first run, CPU-only)
2. Filters COCO detections to class 16 (dog) only
3. Calls `posture.py` per-detection per-frame
4. Writes annotated video (green=standing, red=squatting) and `output/detections.csv`
5. Simulates alerts with configurable cooldown (`alert_cooldown_seconds` in `config.yaml`)

### Evaluation (`evaluate.py`)

Reads detection CSV + ground truth CSV, aggregates multiple detections per frame (highest confidence wins), then computes accuracy, precision, recall, F1, and confusion matrix via scikit-learn.

### Configuration (`config.yaml`)

Key tunable parameters:
- `squat_threshold` — compactness cutoff (lower = more sensitive to squatting)
- `detection_confidence` — YOLO confidence filter
- `alert_cooldown_seconds` — minimum seconds between squat alerts
