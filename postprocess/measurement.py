"""Deterministic post-processing of oracle geometry into numeric answers.

Pure functions over masks and bounding boxes:
  - area from a mask
  - bounding box from a mask
  - maximum diameter from a bounding box (and, optionally, from a mask)
  - object counting
  - circumference (contour perimeter, with a bbox fallback)

Everything works in **pixels** by default. If pixel spacing ``[sx, sy]``
(mm/pixel) is available, ``to_units`` converts lengths to mm and areas to mm^2.

Bounding boxes are ``[x_min, y_min, x_max, y_max]`` where ``x_max``/``y_max`` are
the far edges, so width ``= x_max - x_min`` and height ``= y_max - y_min``.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Mask loading / binarization
# --------------------------------------------------------------------------- #
def load_mask(path: str) -> np.ndarray:
    """Load a mask image as a 2D integer array. Foreground is any nonzero pixel.

    Prefers Pillow (preserves palette / grayscale index values); falls back to
    OpenCV. Multi-channel images are collapsed so that any nonzero channel
    counts as foreground.
    """
    arr = None
    try:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im)
    except Exception:
        arr = None
    if arr is None:
        import cv2

        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"could not read mask: {path}")
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # collapse channels: nonzero in any channel => foreground
        arr = arr[..., : min(3, arr.shape[2])].sum(axis=2)
    return arr


def _binarize(mask: np.ndarray, label: Optional[int] = None) -> np.ndarray:
    if label is None:
        return mask != 0
    return mask == label


# --------------------------------------------------------------------------- #
# Mask measurements
# --------------------------------------------------------------------------- #
def mask_area(mask: np.ndarray, label: Optional[int] = None) -> int:
    """Foreground pixel count (optionally restricted to a specific label value)."""
    return int(np.count_nonzero(_binarize(mask, label)))


def mask_bbox(mask: np.ndarray, label: Optional[int] = None) -> Optional[list]:
    """Tight bounding box of the foreground as ``[x_min, y_min, x_max, y_max]``.

    ``x_max``/``y_max`` are exclusive far edges (so width = x_max - x_min equals
    the pixel extent). Returns None if there is no foreground.
    """
    b = _binarize(mask, label)
    ys, xs = np.nonzero(b)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# alias matching the spec wording ("compute bbox from mask")
def bbox_from_mask(mask: np.ndarray, label: Optional[int] = None) -> Optional[list]:
    return mask_bbox(mask, label)


# --------------------------------------------------------------------------- #
# Bounding-box measurements
# --------------------------------------------------------------------------- #
def bbox_width_height(bbox: Sequence[float]) -> tuple:
    x0, y0, x1, y1 = bbox
    return (abs(x1 - x0), abs(y1 - y0))


def bbox_area(bbox: Sequence[float]) -> float:
    w, h = bbox_width_height(bbox)
    return float(w * h)


def max_diameter_from_bbox(bbox: Sequence[float], mode: str = "diagonal") -> float:
    """Maximum diameter implied by a bounding box.

    ``diagonal`` (default) = the longest distance between two box corners
    (sqrt(w^2 + h^2)); ``max_side`` = max(width, height).
    """
    w, h = bbox_width_height(bbox)
    if mode == "max_side":
        return float(max(w, h))
    return float(math.hypot(w, h))


def bbox_perimeter(bbox: Sequence[float]) -> float:
    w, h = bbox_width_height(bbox)
    return float(2 * (w + h))


def count_objects(items: Sequence) -> int:
    return int(len(items or []))


# --------------------------------------------------------------------------- #
# Optional mask-based shape measurements (need OpenCV; degrade gracefully)
# --------------------------------------------------------------------------- #
def _contours(mask: np.ndarray, label: Optional[int] = None):
    import cv2

    b = _binarize(mask, label).astype("uint8")
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def max_diameter_from_mask(mask: np.ndarray, label: Optional[int] = None) -> float:
    """Max Feret diameter: the largest distance between two foreground boundary
    points. Falls back to the bbox diagonal if OpenCV is unavailable."""
    try:
        cnts = _contours(mask, label)
    except Exception:
        bb = mask_bbox(mask, label)
        return max_diameter_from_bbox(bb) if bb else 0.0
    if not cnts:
        return 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(float)
    if len(pts) == 0:
        return 0.0
    try:
        import cv2

        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    except Exception:
        hull = pts
    best = 0.0
    for i in range(len(hull)):
        diff = hull - hull[i]
        d = float(np.sqrt((diff ** 2).sum(axis=1)).max())
        if d > best:
            best = d
    return best


def contour_perimeter(mask: np.ndarray, label: Optional[int] = None) -> float:
    """Sum of external contour perimeters. Falls back to bbox perimeter without
    OpenCV."""
    try:
        import cv2

        cnts = _contours(mask, label)
    except Exception:
        bb = mask_bbox(mask, label)
        return bbox_perimeter(bb) if bb else 0.0
    return float(sum(cv2.arcLength(c, True) for c in cnts))


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #
def to_units(value_px: float, spacing, kind: str, unit: str) -> float:
    """Convert a pixel-space value to the requested unit.

    ``kind`` is ``"length"`` or ``"area"``. Without spacing (or when ``unit`` is
    ``"pixel"``) the value is returned unchanged. Length uses the mean of
    ``sx``/``sy`` (isotropic assumption); area uses ``sx * sy``.
    """
    if unit in (None, "pixel") or not spacing:
        return float(value_px)
    try:
        sx = float(spacing[0])
        sy = float(spacing[1]) if len(spacing) >= 2 else float(spacing[0])
    except (TypeError, IndexError, ValueError):
        return float(value_px)
    if kind == "area":
        return float(value_px) * sx * sy
    return float(value_px) * ((sx + sy) / 2.0)
