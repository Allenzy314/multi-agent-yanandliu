#!/usr/bin/env python3
"""Repair dataset_detect.json image paths to point at the file's real location.

Why: in some datasets (notably Dataset504_Breast_Nodule) the *detection*
train/test split differs from the *segmentation* split, so a detect entry may say
`imagesTr/X.png` while `X` actually lives in `imagesTs/` (or vice versa). Every
referenced image DOES exist on disk — only the folder prefix is wrong. This
rewrites each entry's folder to match disk (keeping the manifest's bare-name,
no-`_0000` convention) and corrects numTraining/numTest to the real list lengths.

The original manifest is backed up to `dataset_detect.json.orig` before writing.

Usage
-----
  python dataset_tools/fix_detect_paths.py --datasets 504
  python dataset_tools/fix_detect_paths.py            # all datasets (no-op where already correct)
  python dataset_tools/fix_detect_paths.py --datasets 504 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

try:
    from config import DEFAULT_DATA_ROOT
except Exception:  # pragma: no cover
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flare_dataset", "Ultrasound")


def _stem(name: str) -> str:
    """File/ref basename -> stem with .png and any _0000 channel suffix removed."""
    if name.endswith(".png"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def build_lookup(d: str) -> dict[str, str]:
    """stem -> folder ('imagesTr' or 'imagesTs') for every image on disk."""
    lut: dict[str, str] = {}
    for folder in ("imagesTr", "imagesTs"):
        p = os.path.join(d, folder)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.endswith(".png"):
                lut[_stem(f)] = folder
    return lut


def fix_dataset(d: str, dry_run: bool) -> tuple[int, int, int]:
    """Returns (changed, unresolved, total). Writes the file unless dry_run."""
    p = os.path.join(d, "dataset_detect.json")
    if not os.path.exists(p):
        return (0, 0, 0)
    j = json.load(open(p))
    lut = build_lookup(d)
    changed = unresolved = total = 0
    examples: list[str] = []
    for split in ("training", "test"):
        for r in j.get(split, []):
            if not isinstance(r, dict) or "image" not in r:
                continue
            total += 1
            ip = r["image"]
            base = os.path.basename(ip)
            folder = lut.get(_stem(base))
            if folder is None:
                unresolved += 1
                continue
            newpath = f"{folder}/{base}"
            if newpath != ip:
                if len(examples) < 5:
                    examples.append(f"{ip} -> {newpath}")
                r["image"] = newpath
                changed += 1
    # make the count metadata consistent with the actual lists
    old_tr, old_te = j.get("numTraining"), j.get("numTest")
    j["numTraining"] = len(j.get("training", []))
    j["numTest"] = len(j.get("test", []))
    count_fixed = (old_tr, old_te) != (j["numTraining"], j["numTest"])

    name = os.path.basename(d)
    print(f"\n=== {name} ===")
    print(f"  detect entries: {total} | folder fixes: {changed} | unresolved: {unresolved}")
    for e in examples:
        print(f"    {e}")
    if count_fixed:
        print(f"  numTraining {old_tr}->{j['numTraining']}, numTest {old_te}->{j['numTest']}")
    if dry_run:
        print("  (dry-run: nothing written)")
        return (changed, unresolved, total)
    if changed or count_fixed:
        bak = p + ".orig"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
            print(f"  backed up original -> {os.path.basename(bak)}")
        with open(p, "w") as fh:
            json.dump(j, fh, indent=2)
        print("  wrote corrected dataset_detect.json")
    else:
        print("  already correct — no changes")
    return (changed, unresolved, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Only fix these (substring match, e.g. 504)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    folders = sorted(e.path for e in os.scandir(args.data_root)
                     if e.is_dir() and e.name.startswith("Dataset"))
    if args.datasets:
        folders = [f for f in folders if any(s in os.path.basename(f) for s in args.datasets)]
    tot_changed = tot_unres = 0
    for d in folders:
        c, u, _ = fix_dataset(d, args.dry_run)
        tot_changed += c
        tot_unres += u
    print(f"\nTOTAL folder fixes: {tot_changed} | unresolved (truly missing): {tot_unres}")


if __name__ == "__main__":
    main()
