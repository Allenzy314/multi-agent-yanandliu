#!/usr/bin/env python3
"""Reconstruct Dataset508_Fetal_Head/labelsTs from the original Zenodo source.

Dataset508 shipped without test labels (`labelsTs/`). They are recovered from the
source dataset (Zenodo 8265464, Alzubaidi 2023 "Large-Scale Annotation Dataset for
Fetal Head Biometry") whose `*-Segmentation.zip` files contain PASCAL-VOC-style
RGB `SegmentationClass/` masks. The conversion below was VERIFIED to reproduce all
3449 training labels in `labelsTr/` pixel-exact, then applied to the 383 test images.

Per-category rules (reverse-engineered + verified):
  - Trans-ventricular / Trans-cerebellum / Trans-thalamic:
        CSP encoded green (0,255,0); FLARE keeps only {LV, Brain} (drops CSP).
  - Diverse Fetal Head Images (the *_HC images):
        CSP encoded YELLOW (255,255,0); FLARE keeps all {LV, CSP, Brain}.
  FLARE label encoding: background=0, LV=1, CSP=2, Brain=3 (blue=LV, red=Brain).

Prerequisite: the 4 annotated Zenodo zips present in the dataset folder
(Trans-ventricular.zip, Trans-cerebellum.zip, Trans-thalamic.zip,
"Diverse Fetal Head Images.zip"). Download from https://zenodo.org/records/8265464

Usage
-----
  python dataset_tools/rebuild_508_labelsTs.py
  python dataset_tools/rebuild_508_labelsTs.py --verify-train   # re-check 3449/3449
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np
from PIL import Image

try:
    from config import DEFAULT_DATA_ROOT
except Exception:  # pragma: no cover
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flare_dataset", "Ultrasound")

# zip name -> (CSP rgb color, kept FLARE class ids)
RULES = {
    "Trans-ventricular.zip": ((0, 255, 0), {1, 3}),
    "Trans-cerebellum.zip": ((0, 255, 0), {1, 3}),
    "Trans-thalamic.zip": ((0, 255, 0), {1, 3}),
    "Diverse Fetal Head Images.zip": ((255, 255, 0), {1, 2, 3}),
}


def _convert(rgb: np.ndarray, csp: tuple[int, int, int], keep: set[int]) -> np.ndarray:
    out = np.zeros(rgb.shape[:2], np.uint8)
    out[(rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 255)] = 1   # LV blue
    out[(rgb[..., 0] == csp[0]) & (rgb[..., 1] == csp[1]) & (rgb[..., 2] == csp[2])] = 2  # CSP
    out[(rgb[..., 0] == 255) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)] = 3   # Brain red
    out[~np.isin(out, list(keep))] = 0
    return out


def _open_segmaps(d: str):
    """Return {stem: (rgb_array_loader, csp, keep)} across all 4 source zips."""
    lut = {}
    for zn, (csp, keep) in RULES.items():
        zp = os.path.join(d, zn)
        if not os.path.exists(zp):
            raise SystemExit(f"missing source zip: {zp}\n  download from "
                             "https://zenodo.org/records/8265464")
        outer = zipfile.ZipFile(zp)
        seg = [n for n in outer.namelist()
               if n.endswith(".zip") and "segmentation" in n.lower()][0]
        inner = zipfile.ZipFile(io.BytesIO(outer.read(seg)))
        for n in inner.namelist():
            if "SegmentationClass/" in n and n.endswith(".png"):
                lut[os.path.basename(n)[:-4]] = (inner, n, csp, keep)
    return lut


def _label_for(lut, stem: str) -> np.ndarray | None:
    if stem not in lut:
        return None
    inner, n, csp, keep = lut[stem]
    src = np.asarray(Image.open(io.BytesIO(inner.read(n))))
    if src.ndim != 3:
        src = np.stack([src] * 3, -1)
    return _convert(src[..., :3], csp, keep)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--verify-train", action="store_true",
                    help="Re-verify the rules reproduce all labelsTr pixel-exact, then exit")
    args = ap.parse_args()
    d = os.path.join(args.data_root, "Dataset508_Fetal_Head")
    lut = _open_segmaps(d)

    if args.verify_train:
        ok = tot = 0
        for f in os.listdir(os.path.join(d, "labelsTr")):
            if not f.endswith(".png"):
                continue
            tot += 1
            lab = _label_for(lut, f[:-4])
            fl = np.asarray(Image.open(os.path.join(d, "labelsTr", f)))
            if lab is not None and lab.shape == fl.shape and np.array_equal(lab, fl):
                ok += 1
        print(f"training reconstruction: {ok}/{tot} pixel-exact")
        return

    out = os.path.join(d, "labelsTs")
    os.makedirs(out, exist_ok=True)
    test = [f[:-9] for f in os.listdir(os.path.join(d, "imagesTs")) if f.endswith("_0000.png")]
    wrote = missing = 0
    for stem in test:
        lab = _label_for(lut, stem)
        if lab is None:
            missing += 1
            continue
        img = Image.open(os.path.join(d, "imagesTs", stem + "_0000.png"))
        if (lab.shape[1], lab.shape[0]) != img.size:
            lab = np.asarray(Image.fromarray(lab).resize(img.size, Image.NEAREST))
        Image.fromarray(lab.astype(np.uint8)).save(os.path.join(out, stem + ".png"))
        wrote += 1
    print(f"wrote labelsTs={wrote} (missing from source: {missing}) -> {out}")


if __name__ == "__main__":
    main()
