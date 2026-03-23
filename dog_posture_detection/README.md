# Dog Posture Detection

Detects when a dog is squatting (as if going to the bathroom) in video footage using YOLOv8n object detection and bounding box aspect ratio analysis.

## Setup

### Windows

```powershell
git clone https://github.com/<your-username>/dog-posture-detection.git
cd dog-posture-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### WSL / Ubuntu

```bash
# 1. Install system dependencies (needed for OpenCV and general build tools)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git \
    libgl1-mesa-glx libglib2.0-0

# 2. Clone the repo
git clone https://github.com/<your-username>/dog-posture-detection.git
cd dog-posture-detection

# 3. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt
```

> **WSL display note:** The interactive viewer (`viewer.py`) opens an OpenCV GUI
> window. For this to work in WSL you need an X server or WSLg (Windows 11 ships
> with WSLg by default). If you're on Windows 10 you'll need an X server like
> [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or
> [X410](https://x410.dev/), then run `export DISPLAY=:0` before launching the
> viewer. Detection and evaluation run headless and work without any display setup.

The YOLOv8n model weights (`yolov8n.pt`) are downloaded automatically on first run.

## Usage

### 1. Run Detection

```bash
python detect.py path/to/video.mp4
python detect.py /home/metal/DoggyDont/Doggydont_videos
```

Options:
- `--threshold 0.75` — override the squat compactness threshold
- `--config config.yaml` — use a custom config file
- `--no-video` — skip saving the annotated output video

Output (saved to `output/`):
- `annotated_output.avi` — video with color-coded bounding boxes
- `detections.csv` — per-frame detection log

> **WSL file path tip:** You can reference videos on your Windows filesystem from
> WSL via `/mnt/c/Users/<you>/...`. For best I/O performance, copy the video into
> the WSL filesystem first:
> ```bash
> cp /mnt/c/Users/<you>/Videos/dog_clip.mp4 ./
> python detect.py dog_clip.mp4
> ```

### 2. View Results

```bash
python viewer.py                         # plays output/annotated_output.avi
python viewer.py path/to/annotated.avi   # plays a specific file
```

Controls:
| Key | Action |
|-----|--------|
| SPACE | Pause / Resume |
| D | Step forward one frame (paused) |
| A | Step backward one frame (paused) |
| Q / ESC | Quit |

> **WSL:** Requires a working display (WSLg or X server). If you only need to
> review results without the interactive viewer, open the annotated `.avi` file
> directly in Windows — it's in the `output/` folder.

### 3. Evaluate Against Ground Truth

Create a CSV with your labeled frames:

```csv
frame_number,true_label
0,STANDING
30,STANDING
60,SQUATTING
90,SQUATTING
120,STANDING
```

Run evaluation:

```bash
python evaluate.py --ground-truth labels.csv
python evaluate.py --ground-truth labels.csv --detections output/detections.csv
```

This prints accuracy, precision, recall, F1, and a confusion matrix.

### 4. Tune the Threshold

The camera is mounted overhead or at a steep downward angle. The dog can face **any direction** in the frame. The classifier uses an orientation-invariant **compactness** ratio:

```
compactness = minor_axis / major_axis
            = min(bbox_h, bbox_w) / max(bbox_h, bbox_w)
```

The longer bounding box side is always treated as the dog's body-length axis, the shorter as the cross-body axis — regardless of whether the dog is walking left-right, up-down, or diagonally.

- **Low compactness** (~0.25–0.50) when standing — the dog is elongated
- **High compactness** (~0.65–0.90) when squatting — the legs splay outward, making the box more square

The threshold is the cutoff: compactness **>=** threshold = SQUATTING, compactness **<** threshold = STANDING.

Experiment:

```bash
python detect.py video.mp4 --threshold 0.70
python detect.py video.mp4 --threshold 0.80
```

Then evaluate each run against your ground truth to find the best value.

## Configuration

Edit `config.yaml` to change defaults:

```yaml
squat_threshold: 0.75          # compactness ratio cutoff
detection_confidence: 0.3      # YOLO confidence filter
alert_cooldown_seconds: 3.0    # min seconds between alerts
output_dir: "output"
```

## How It Works

1. **Frame extraction** — OpenCV reads each frame from the input video
2. **Dog detection** — YOLOv8n detects objects; only COCO class 16 (dog) is kept
3. **Posture classification** — orientation-invariant compactness ratio (minor_axis / major_axis) is compared to the threshold: at or above = SQUATTING, below = STANDING. Works regardless of which direction the dog faces in the overhead frame.
4. **Alert simulation** — SQUATTING detections trigger a timestamped console alert (with cooldown to avoid spam)
5. **Output** — annotated video (green=standing, red=squatting) and CSV log

## Project Structure

```
dog-posture-detection/
├── detect.py                # Main detection pipeline
├── posture.py               # Orientation-invariant posture classifier
├── evaluate.py              # Score detections against ground truth labels
├── viewer.py                # Interactive annotated video viewer
├── config.yaml              # Tunable settings (threshold, cooldown, paths)
├── requirements.txt         # Python dependencies
├── sample_ground_truth.csv  # Template for labeling ground truth
├── README.md
└── output/                  # Detection outputs (created at runtime)
```

## Requirements

- Python 3.10+
- Runs on CPU (no GPU required)
- ~6 MB model download on first run
- **WSL:** Ubuntu 20.04+ with `libgl1-mesa-glx` and `libglib2.0-0`
