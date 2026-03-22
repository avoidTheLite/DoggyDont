"""
Evaluation script — scores detection results against labeled ground truth.

Expects a ground truth CSV with columns: frame_number, true_label
where true_label is SQUATTING or STANDING.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


def evaluate(detection_csv: str, ground_truth_csv: str):
    detections = pd.read_csv(detection_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    # Aggregate detections to one posture per frame (take the most confident detection)
    if detections.empty:
        print("No detections found in the detection log.")
        return

    best_per_frame = (
        detections.sort_values("confidence", ascending=False)
        .drop_duplicates(subset="frame_number", keep="first")
    )

    merged = ground_truth.merge(
        best_per_frame[["frame_number", "posture"]],
        on="frame_number",
        how="inner",
    )

    if merged.empty:
        print("No matching frames between detections and ground truth.")
        return

    y_true = merged["true_label"].str.upper()
    y_pred = merged["posture"].str.upper()

    labels = ["STANDING", "SQUATTING"]

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Matched frames: {len(merged)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for i, label in enumerate(labels):
        print(f"  {label:12s}  precision={precision[i]:.3f}  "
              f"recall={recall[i]:.3f}  f1={f1[i]:.3f}  n={support[i]}")
    print()

    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>14s} {'STANDING':>10s} {'SQUATTING':>10s}")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, label in enumerate(labels):
        print(f"  {label:>12s} {cm[i][0]:>10d} {cm[i][1]:>10d}")
    print()

    print("Full classification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate dog posture detections")
    parser.add_argument("--detections", default="output/detections.csv",
                        help="Path to detection log CSV (from detect.py)")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to ground truth CSV (frame_number, true_label)")
    args = parser.parse_args()

    evaluate(args.detections, args.ground_truth)


if __name__ == "__main__":
    main()
