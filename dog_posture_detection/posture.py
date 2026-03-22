"""
Posture classifier module — orientation-invariant.

Classifies dog posture from bounding box geometry. Designed for an overhead /
steep downward camera angle where the dog can face ANY direction in the frame.

Instead of using image-axis-aligned height/width (which changes meaning when
the dog rotates), we define a body-relative coordinate system:

  major_axis = max(bbox_height, bbox_width)   — the dog's lengthwise dimension
  minor_axis = min(bbox_height, bbox_width)   — the dog's crosswise dimension

  compactness = minor_axis / major_axis       — always in (0, 1]

This ratio is orientation-invariant:

  - A STANDING dog is elongated       -> low  compactness  (e.g. 0.25–0.50)
  - A SQUATTING dog splays its legs   -> high compactness  (approaches 1.0)

Therefore: compactness >= threshold  =>  SQUATTING
           compactness <  threshold  =>  STANDING
"""

from dataclasses import dataclass

# Default threshold — tune via config.yaml or --threshold CLI flag
DEFAULT_SQUAT_THRESHOLD = 0.75


@dataclass
class PostureResult:
    """Result of a single posture classification."""
    compactness: float    # minor_axis / major_axis, orientation-invariant
    major_axis: float     # longer bbox dimension (body length direction)
    minor_axis: float     # shorter bbox dimension (cross-body direction)
    posture: str          # "SQUATTING" or "STANDING"
    threshold_used: float


def compute_compactness(bbox_height: float, bbox_width: float) -> tuple[float, float, float]:
    """
    Compute the orientation-invariant compactness ratio.

    Treats the longer bounding box dimension as the dog's lengthwise (major)
    axis and the shorter as the crosswise (minor) axis, regardless of which
    image axis they correspond to.

    Returns:
        (compactness, major_axis, minor_axis)
        compactness is minor / major, in the range (0, 1].
        Returns (0.0, 0.0, 0.0) if both dimensions are zero.
    """
    major = max(bbox_height, bbox_width)
    minor = min(bbox_height, bbox_width)
    if major <= 0:
        return 0.0, 0.0, 0.0
    return minor / major, major, minor


def classify_posture(
    bbox_height: float,
    bbox_width: float,
    threshold: float = DEFAULT_SQUAT_THRESHOLD,
) -> PostureResult:
    """
    Classify a dog's posture from its bounding box dimensions.

    Orientation-invariant: the dog can face any direction in the overhead frame.
    The longer bbox side is treated as the body's lengthwise axis, the shorter
    as the crosswise axis.

    A standing dog is elongated (low compactness).
    A squatting dog splays outward, becoming more square (high compactness).

    Returns SQUATTING when compactness >= threshold, STANDING otherwise.

    Args:
        bbox_height: Bounding box height in pixels (image y-axis).
        bbox_width:  Bounding box width in pixels (image x-axis).
        threshold:   Compactness ratio at or above which posture is SQUATTING.

    Returns:
        PostureResult with compactness, axis lengths, posture label, and threshold.
    """
    compactness, major, minor = compute_compactness(bbox_height, bbox_width)
    posture = "SQUATTING" if compactness >= threshold else "STANDING"
    return PostureResult(
        compactness=compactness,
        major_axis=major,
        minor_axis=minor,
        posture=posture,
        threshold_used=threshold,
    )
