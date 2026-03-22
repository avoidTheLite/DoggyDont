"""
Dog Posture Detection — Main detection script.

Processes a video file, detects dogs via YOLOv8n, classifies posture
(SQUATTING vs STANDING) by bounding box aspect ratio, and logs results.

Camera assumption: overhead / steep downward angle.
  - Standing dog = elongated, low aspect ratio
  - Squatting dog = legs splay outward, ratio increases toward square
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from posture import classify_posture

# COCO class index for "dog"
DOG_CLASS_ID = 16


def load_config(config_path: str = "config.yaml") -> dict:
    default = {
        "squat_threshold": 0.75,
        "detection_confidence": 0.3,
        "alert_cooldown_seconds": 3.0,
        "output_dir": "output",
        "output_video_filename": "annotated_output.avi",
        "detection_log_filename": "detections.csv",
        "display_window": True,
        "window_width": 960,
        "window_height": 540,
    }
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        default.update(loaded)
    return default


def run_detection(video_path: str, threshold: float | None = None,
                  config_path: str = "config.yaml", save_video: bool = True):
    config = load_config(config_path)
    if threshold is not None:
        config["squat_threshold"] = threshold

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        out_path = str(output_dir / config["output_video_filename"])
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    log_path = str(output_dir / config["detection_log_filename"])
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["frame_number", "timestamp", "x1", "y1", "x2", "y2",
                         "confidence", "compactness", "major_axis", "minor_axis",
                         "posture"])

    squat_threshold = config["squat_threshold"]
    det_conf = config["detection_confidence"]
    cooldown = config["alert_cooldown_seconds"]
    last_alert_time = 0.0

    # Summary stats
    total_detections = 0
    squat_events = 0
    confidence_sum = 0.0

    print(f"Processing: {video_path}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}, Resolution: {width}x{height}")
    print(f"  Squat threshold: {squat_threshold}")
    print(f"  Detection confidence: {det_conf}")
    print("-" * 60)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_num / fps
        results = model(frame, conf=det_conf, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id != DOG_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bw = x2 - x1
                bh = y2 - y1
                result = classify_posture(bh, bw, squat_threshold)
                compactness = result.compactness
                posture = result.posture

                total_detections += 1
                confidence_sum += conf

                csv_writer.writerow([
                    frame_num, f"{timestamp:.3f}",
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{conf:.3f}", f"{compactness:.3f}",
                    f"{result.major_axis:.1f}", f"{result.minor_axis:.1f}",
                    posture
                ])

                # Alert simulator
                if posture == "SQUATTING":
                    squat_events += 1
                    now = time.time()
                    if now - last_alert_time >= cooldown:
                        print(f"  ⚠ SQUAT ALERT  frame={frame_num}  "
                              f"time={timestamp:.2f}s  compact={compactness:.3f}  "
                              f"conf={conf:.2f}")
                        last_alert_time = now

                # Annotate frame
                color = (0, 0, 255) if posture == "SQUATTING" else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{posture} {compactness:.2f}"
                label_y = max(int(y1) - 8, 16)
                cv2.putText(frame, label, (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer:
            writer.write(frame)

        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames...")

    cap.release()
    if writer:
        writer.release()
    log_file.close()

    # Summary
    avg_conf = confidence_sum / total_detections if total_detections else 0.0
    print("-" * 60)
    print("Summary:")
    print(f"  Total frames processed: {frame_num}")
    print(f"  Dog detections: {total_detections}")
    print(f"  Squat detections: {squat_events}")
    print(f"  Avg detection confidence: {avg_conf:.3f}")
    if save_video:
        print(f"  Annotated video: {output_dir / config['output_video_filename']}")
    print(f"  Detection log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Dog Posture Detection")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Squat aspect-ratio threshold (overrides config)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving annotated output video")
    args = parser.parse_args()

    run_detection(args.video, threshold=args.threshold,
                  config_path=args.config, save_video=not args.no_video)


if __name__ == "__main__":
    main()
