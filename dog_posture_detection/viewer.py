"""
Results Viewer — plays annotated video with pause/step-through controls.

Controls:
  SPACE  — pause / resume
  D      — step forward one frame (while paused)
  A      — step backward one frame (while paused)
  Q/ESC  — quit
"""

import argparse
from pathlib import Path

import cv2
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    default = {"window_width": 960, "window_height": 540}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        default.update(loaded)
    return default


def view(video_path: str, config_path: str = "config.yaml"):
    config = load_config(config_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps)
    win_w = config["window_width"]
    win_h = config["window_height"]
    paused = False
    frame_idx = 0

    window_name = "Dog Posture Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)

    print(f"Playing: {video_path}")
    print(f"  SPACE=pause/resume  D=step forward  A=step back  Q/ESC=quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # HUD overlay
        display = frame.copy()
        hud = f"Frame {frame_idx}/{total_frames}"
        if paused:
            hud += "  [PAUSED]"
        cv2.putText(display, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("d") and paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == ord("a") and paused:
            new_pos = max(0, frame_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="View annotated detection video")
    parser.add_argument("video", nargs="?", default="output/annotated_output.avi",
                        help="Path to annotated video (default: output/annotated_output.avi)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    view(args.video, config_path=args.config)


if __name__ == "__main__":
    main()
