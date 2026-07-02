#!/usr/bin/env python3
"""Integrity / completeness checker for the FLARE25 Ultrasound datasets.

For every DatasetXXX_* folder under the Ultrasound root it validates the JSON
manifests against what is actually on disk:

  * dataset.json        (segmentation): every training/test image+label exists,
                        counts match numTraining/numTest, no duplicate refs.
  * dataset_detect.json (detection):    every referenced image exists (resolving
                        the `_0000` channel suffix), boxes are well-formed,
                        class ids are within the label set.
  * dataset_cls.json    (classification): every image exists, label is a valid
                        class index.
  * disk pairing:       imagesTr<->labelsTr (and imagesTs<->labelsTs) line up;
                        reports orphan files and image/label count mismatches.

With --deep it also opens the PNGs (readability) and checks that label pixel
values fall within the declared label set (sampled unless --deep-full).

Exit code is non-zero if any ERROR-level problem is found.

Usage
-----
  python dataset_tools/check_integrity.py
  python dataset_tools/check_integrity.py --datasets 509 510
  python dataset_tools/check_integrity.py --deep              # sampled pixel checks
  python dataset_tools/check_integrity.py --deep-full         # every file (slow)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

try:
    from config import DEFAULT_DATA_ROOT
except Exception:  # pragma: no cover - allow running outside the package dir
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "flare_dataset", "Ultrasound")

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
_COLOR = {ERROR: "\033[31m", WARN: "\033[33m", INFO: "\033[36m"}
_RESET = "\033[0m"


class Report:
    """Collects severity-tagged issues for one dataset."""

    def __init__(self, name: str):
        self.name = name
        self.issues: list[tuple[str, str]] = []
        self.notes: list[str] = []

    def add(self, sev: str, msg: str) -> None:
        self.issues.append((sev, msg))

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def errors(self) -> int:
        return sum(1 for s, _ in self.issues if s == ERROR)

    @property
    def warns(self) -> int:
        return sum(1 for s, _ in self.issues if s == WARN)

    @property
    def status(self) -> str:
        return ERROR if self.errors else (WARN if self.warns else "OK")


def _exists(root: str, rel: str) -> bool:
    return os.path.exists(os.path.join(root, rel))


def _resolve_image(root: str, rel: str) -> str | None:
    """Detect/cls manifests drop the `_0000` channel suffix; resolve either form."""
    if _exists(root, rel):
        return rel
    stem, ext = os.path.splitext(rel)
    alt = f"{stem}_0000{ext}"
    return alt if _exists(root, alt) else None


def _fmt_missing(missing: list[str], total: int, kind: str) -> str:
    head = ", ".join(missing[:5])
    more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
    return f"{len(missing)}/{total} {kind} missing: {head}{more}"


def _listdir(root: str, sub: str) -> list[str] | None:
    p = os.path.join(root, sub)
    return sorted(os.listdir(p)) if os.path.isdir(p) else None


# ---------------------------------------------------------------------------
# manifest checks
# ---------------------------------------------------------------------------
def check_seg(d: str, rep: Report) -> None:
    p = os.path.join(d, "dataset.json")
    if not os.path.exists(p):
        rep.add(WARN, "dataset.json (segmentation) not found")
        return
    try:
        j = json.load(open(p))
    except Exception as exc:  # noqa: BLE001
        rep.add(ERROR, f"dataset.json is not valid JSON: {exc}")
        return
    labels = set((j.get("labels") or {}).values())
    for split, count_key in (("training", "numTraining"), ("test", "numTest")):
        recs = j.get(split, [])
        declared = j.get(count_key)
        if declared is not None and declared != len(recs):
            rep.add(WARN, f"dataset.json {split}: {count_key}={declared} "
                          f"but list has {len(recs)}")
        # nnU-Net/FLARE convention: training entries are {image,label} dicts;
        # the test split is a list of image-path strings (labels withheld).
        miss_img, miss_lab, seen, malformed = [], [], set(), 0
        for r in recs:
            if isinstance(r, str):
                img, lab = r, None
            elif isinstance(r, dict):
                img, lab = r.get("image"), r.get("label")
            else:
                malformed += 1
                continue
            if img in seen:
                rep.add(WARN, f"dataset.json {split}: duplicate image ref {img}")
            seen.add(img)
            if img and not _exists(d, img):
                miss_img.append(img)
            if lab and not _exists(d, lab):
                miss_lab.append(lab)
        if miss_img:
            rep.add(ERROR, "dataset.json " + split + " " + _fmt_missing(miss_img, len(recs), "images"))
        if miss_lab:
            rep.add(ERROR, "dataset.json " + split + " " + _fmt_missing(miss_lab, len(recs), "labels"))
        if malformed:
            rep.add(WARN, f"dataset.json {split}: {malformed} entries are not "
                          f"{{image,label}} objects")
        if recs and not miss_img and not miss_lab and not malformed:
            rep.note(f"seg {split}: {len(recs)} image+label refs OK")
    rep._seg = j  # stash for deep check


def check_detect(d: str, rep: Report) -> None:
    p = os.path.join(d, "dataset_detect.json")
    if not os.path.exists(p):
        return
    try:
        j = json.load(open(p))
    except Exception as exc:  # noqa: BLE001
        rep.add(ERROR, f"dataset_detect.json invalid JSON: {exc}")
        return
    n_labels = len(j.get("labels") or {})
    for split, count_key in (("training", "numTraining"), ("test", "numTest")):
        recs = j.get(split, [])
        declared = j.get(count_key)
        if declared is not None and declared != len(recs):
            rep.add(WARN, f"dataset_detect.json {split}: {count_key}={declared} "
                          f"but list has {len(recs)}")
        miss, bad_boxes = [], 0
        for r in recs:
            if not isinstance(r, dict):
                bad_boxes += 1
                continue
            if _resolve_image(d, r.get("image", "")) is None:
                miss.append(r.get("image"))
            for b in r.get("boxes", []):
                cls = b.get("class")
                pts = b.get("points")
                ok = (isinstance(cls, int) and (n_labels == 0 or 0 <= cls < n_labels)
                      and isinstance(pts, list) and len(pts) >= 3
                      and all(isinstance(pt, list) and len(pt) == 2 for pt in pts))
                if not ok:
                    bad_boxes += 1
        if miss:
            rep.add(ERROR, "dataset_detect.json " + split + " " + _fmt_missing(miss, len(recs), "images"))
        if bad_boxes:
            rep.add(WARN, f"dataset_detect.json {split}: {bad_boxes} malformed boxes "
                          f"(bad class id or points)")
        if recs and not miss:
            rep.note(f"detect {split}: {len(recs)} image refs OK")


def check_cls(d: str, rep: Report) -> None:
    p = os.path.join(d, "dataset_cls.json")
    if not os.path.exists(p):
        return
    try:
        j = json.load(open(p))
    except Exception as exc:  # noqa: BLE001
        rep.add(ERROR, f"dataset_cls.json invalid JSON: {exc}")
        return
    classes = set((j.get("labels") or {}).values())
    for split in ("training", "test"):
        recs = j.get(split, [])
        miss, bad = [], 0
        for r in recs:
            if not isinstance(r, dict):
                bad += 1
                continue
            if _resolve_image(d, r.get("image", "")) is None:
                miss.append(r.get("image"))
            lab = r.get("label")
            if classes and (not isinstance(lab, int) or lab not in classes):
                bad += 1
        if miss:
            rep.add(ERROR, "dataset_cls.json " + split + " " + _fmt_missing(miss, len(recs), "images"))
        if bad:
            rep.add(WARN, f"dataset_cls.json {split}: {bad} labels outside {sorted(classes)}")
        if recs and not miss and not bad:
            rep.note(f"cls {split}: {len(recs)} entries OK")


# ---------------------------------------------------------------------------
# disk-level checks
# ---------------------------------------------------------------------------
def _stems(files: list[str], is_image: bool) -> set[str]:
    out = set()
    for f in files:
        if not f.lower().endswith(".png"):
            continue
        stem = f[:-4]
        if is_image and stem.endswith("_0000"):
            stem = stem[:-5]
        out.add(stem)
    return out


def check_disk_pairing(d: str, rep: Report) -> None:
    for img_sub, lab_sub in (("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")):
        imgs = _listdir(d, img_sub)
        labs = _listdir(d, lab_sub)
        if imgs is None and labs is None:
            continue
        if imgs is None:
            rep.add(ERROR, f"{img_sub}/ directory missing")
            continue
        if labs is None:
            # only an error if the seg manifest expects labels for this split
            rep.add(WARN, f"{lab_sub}/ directory missing ({len(imgs)} images have no labels on disk)")
            continue
        si, sl = _stems(imgs, True), _stems(labs, False)
        img_no_lab, lab_no_img = si - sl, sl - si
        if img_no_lab:
            rep.add(WARN, f"{img_sub}: {len(img_no_lab)} image(s) without a matching "
                          f"{lab_sub} label, e.g. {sorted(img_no_lab)[:3]}")
        if lab_no_img:
            rep.add(WARN, f"{lab_sub}: {len(lab_no_img)} orphan label(s) without a "
                          f"matching {img_sub} image, e.g. {sorted(lab_no_img)[:3]}")
        rep.note(f"{img_sub}={len(imgs)} {lab_sub}={len(labs)}")


def check_leftovers(d: str, rep: Report) -> None:
    """Flag leftover raw archives/folders (not an integrity error, just clutter)."""
    big = []
    for entry in os.scandir(d):
        low = entry.name.lower()
        if low.endswith((".zip", ".rar", ".7z", ".tar", ".gz")):
            try:
                big.append((entry.name, entry.stat().st_size))
            except OSError:
                pass
    if big:
        total = sum(s for _, s in big) / 1e9
        rep.add(INFO, f"{len(big)} leftover archive(s) (~{total:.1f} GB), e.g. "
                      f"{sorted(n for n, _ in big)[:3]}")


# ---------------------------------------------------------------------------
# deep (pixel) checks
# ---------------------------------------------------------------------------
def check_deep(d: str, rep: Report, full: bool, sample: int) -> None:
    j = getattr(rep, "_seg", None)
    if not j:
        return
    import numpy as np
    from PIL import Image
    allowed = set((j.get("labels") or {}).values())
    bad_img, bad_lab_open, out_of_range = [], [], []
    for split in ("training", "test"):
        recs = j.get(split, [])
        if not full and len(recs) > sample:
            step = max(1, len(recs) // sample)
            recs = recs[::step][:sample]
        for r in recs:
            if isinstance(r, str):
                img_ref, lab_ref = r, None
            elif isinstance(r, dict):
                img_ref, lab_ref = r.get("image", ""), r.get("label")
            else:
                continue
            ip = os.path.join(d, img_ref)
            lp = os.path.join(d, lab_ref) if lab_ref else None
            if os.path.exists(ip):
                try:
                    Image.open(ip).load()
                except Exception:  # noqa: BLE001
                    bad_img.append(img_ref)
            if lp and os.path.exists(lp):
                try:
                    arr = np.asarray(Image.open(lp))
                    extra = set(np.unique(arr).tolist()) - allowed
                    if extra:
                        out_of_range.append((lab_ref, sorted(extra)))
                except Exception:  # noqa: BLE001
                    bad_lab_open.append(lab_ref)
    scope = "all" if full else f"~{sample}/split"
    if bad_img:
        rep.add(ERROR, f"[deep {scope}] {len(bad_img)} unreadable image(s), e.g. {bad_img[:3]}")
    if bad_lab_open:
        rep.add(ERROR, f"[deep {scope}] {len(bad_lab_open)} unreadable label(s), e.g. {bad_lab_open[:3]}")
    if out_of_range:
        ex = out_of_range[:3]
        rep.add(ERROR, f"[deep {scope}] {len(out_of_range)} label(s) with values outside "
                       f"{sorted(allowed)}, e.g. {ex}")
    if not (bad_img or bad_lab_open or out_of_range):
        rep.note(f"deep ({scope}): pixels readable, label values within {sorted(allowed)}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def check_dataset(d: str, deep: bool, deep_full: bool, sample: int) -> Report:
    rep = Report(os.path.basename(d))
    # nothing populated at all?
    if not any(os.path.isdir(os.path.join(d, s)) for s in ("imagesTr", "imagesTs")):
        rep.add(ERROR, "no imagesTr/ or imagesTs/ — dataset is not populated")
    check_seg(d, rep)
    check_detect(d, rep)
    check_cls(d, rep)
    check_disk_pairing(d, rep)
    check_leftovers(d, rep)
    if deep or deep_full:
        check_deep(d, rep, deep_full, sample)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Only check these (match by substring, e.g. 509 510)")
    ap.add_argument("--deep", action="store_true",
                    help="Also open PNGs and check label pixel values (sampled)")
    ap.add_argument("--deep-full", action="store_true",
                    help="Deep check on EVERY file (slow)")
    ap.add_argument("--sample", type=int, default=200,
                    help="Deep-check sample size per split (default 200)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    root = args.data_root
    if not os.path.isdir(root):
        sys.exit(f"data-root not found: {root}")
    folders = sorted(e.path for e in os.scandir(root)
                     if e.is_dir() and e.name.startswith("Dataset"))
    if args.datasets:
        folders = [f for f in folders if any(s in os.path.basename(f) for s in args.datasets)]
    if not folders:
        sys.exit("no matching Dataset* folders")

    def color(sev: str, text: str) -> str:
        if args.no_color or not sys.stdout.isatty():
            return text
        return f"{_COLOR.get(sev, '')}{text}{_RESET}"

    reports = []
    for d in folders:
        rep = check_dataset(d, args.deep, args.deep_full, args.sample)
        reports.append(rep)
        badge = color(rep.status, f"[{rep.status}]")
        print(f"\n=== {rep.name}  {badge} ===")
        for n in rep.notes:
            print(f"  · {n}")
        for sev, msg in rep.issues:
            print(f"  {color(sev, sev):<6} {msg}")

    # summary
    print("\n" + "=" * 60)
    print(f"{'DATASET':<34} {'STATUS':<7} ERR WARN")
    tot_e = tot_w = 0
    for rep in reports:
        tot_e += rep.errors
        tot_w += rep.warns
        print(f"{rep.name:<34} {color(rep.status, rep.status):<7} "
              f"{rep.errors:>3} {rep.warns:>4}")
    print("-" * 60)
    print(f"{'TOTAL':<34} {'':<7} {tot_e:>3} {tot_w:>4}")
    print(f"\n{len(reports)} datasets | {tot_e} errors | {tot_w} warnings")
    sys.exit(1 if tot_e else 0)


if __name__ == "__main__":
    main()
