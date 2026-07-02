"""Generate a tiny synthetic unified-annotation dataset for tests and smoke runs.

Produces, under ``--out``::

    annotation.json          # list of unified records
    images/case_XXXX.png     # grayscale "ultrasound" backdrop with a shape
    masks/case_XXXX_<name>_<k>.png   # binary mask per object (255 = foreground)

Deterministic: shapes are a function of the case index (no randomness), so the
oracle ground truth is exact and repeatable. Paths in the annotation are
relative to the annotation file, so ``AnnotationStore.from_file`` resolves them
with no extra ``--image_dir``.

Usage:
    python scripts/make_synthetic_fixture.py --out /tmp/fixture --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

TARGET_NAME = "thyroid_nodule"
CLASSES = ["benign", "malignant"]


def _ellipse_bbox(w, h, i, k):
    """Deterministic ellipse bbox within a (w, h) image for object k of case i."""
    cx = int(w * (0.30 + 0.15 * ((i + 2 * k) % 3)))
    cy = int(h * (0.30 + 0.12 * ((i + k) % 4)))
    rw = int(w * (0.10 + 0.04 * ((i + k) % 3)) + 6)
    rh = int(h * (0.09 + 0.05 * ((i + 2 * k) % 3)) + 6)
    x0 = max(1, cx - rw)
    y0 = max(1, cy - rh)
    x1 = min(w - 1, cx + rw)
    y1 = min(h - 1, cy + rh)
    return [int(x0), int(y0), int(x1), int(y1)]


def build(out_dir: str, n: int, width: int = 160, height: int = 128):
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    records = []
    for i in range(n):
        cid = f"case_{i:06d}"
        # backdrop: mild gradient so it looks vaguely like an image
        base = np.tile(np.linspace(20, 90, width, dtype=np.uint8), (height, 1))
        img = Image.fromarray(base, mode="L")
        draw = ImageDraw.Draw(img)

        n_obj = 2 if (i % 3 == 0) else 1
        objects = []
        for k in range(n_obj):
            bbox = _ellipse_bbox(width, height, i, k)
            # draw the nodule on the image
            draw.ellipse(bbox, fill=200)
            # binary mask for this object
            mask_img = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_img).ellipse(bbox, fill=255)
            mask_rel = os.path.join("masks", f"{cid}_{TARGET_NAME}_{k}.png")
            mask_img.save(os.path.join(out_dir, mask_rel))
            objects.append(
                {
                    "name": TARGET_NAME,
                    "class_label": CLASSES[i % len(CLASSES)],
                    "bbox": bbox,
                    "mask_path": mask_rel,
                    "contour": None,
                }
            )

        img_rel = os.path.join("images", f"{cid}.png")
        img.save(os.path.join(out_dir, img_rel))
        records.append(
            {
                "case_id": cid,
                "dataset": "synthetic_thyroid_nodule",
                "modality": "ultrasound",
                "image_path": img_rel,
                "image_size": [width, height],
                "spacing": None,
                "objects": objects,
                "image_label": CLASSES[i % len(CLASSES)],
                "measurements": {},
            }
        )

    ann_path = os.path.join(out_dir, "annotation.json")
    with open(ann_path, "w") as f:
        json.dump(records, f, indent=2)
    return ann_path


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic unified-annotation fixture.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=8, help="number of cases")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ann = build(args.out, args.n, args.width, args.height)
    print(f"wrote {ann} ({args.n} cases)")


if __name__ == "__main__":
    main()
