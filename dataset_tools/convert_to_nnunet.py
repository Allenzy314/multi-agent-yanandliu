#!/usr/bin/env python3
"""Best-effort conversion of the raw sources into an nnU-Net-style layout.

For each dataset it scans  <DatasetXXX>/raw_source/  and writes:
    <DatasetXXX>/imagesTr_generated/<stem>_0000.png   (grayscale image)
    <DatasetXXX>/labelsTr_generated/<stem>.png        (label map, if available)
    <DatasetXXX>/dataset_generated.json               (our manifest)

IMPORTANT — read this
---------------------
The official dataset.json describes the organizers' *pre-sliced* layout
(17264 / 4503 / 2678 PNGs with specific patient/frame filenames). The public
sources are the *raw* datasets (NIfTI volumes, echo videos, original masks).
The FLARE preprocessing scripts that produce the exact slicing/naming are not
public, so this converter CANNOT reproduce those filenames byte-for-byte.
It produces a clean, usable nnU-Net dataset from whatever the raw sources
contain, written to *_generated/ folders so the official jsons are untouched.

Always run `--inspect` first to see the real raw layout, then convert.

Usage
-----
  python dataset_tools/convert_to_nnunet.py --inspect
  python dataset_tools/convert_to_nnunet.py --datasets 509 510
  python dataset_tools/convert_to_nnunet.py --datasets 510 --limit 20   # quick test

(511 / Fetal_NT is excluded: its FUSEP source was removed from Kaggle / HTTP 403.)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np

from config import AVAILABLE_KEYS, DEFAULT_DATA_ROOT, SOURCES, dataset_dir, raw_dir

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VID_EXT = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
NII_EXT = {".nii", ".gz"}  # .nii or .nii.gz
LABEL_HINTS = ("_gt", "label", "mask", "seg", "annotation", "_lab")


def _log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def _ext(path: str) -> str:
    p = path.lower()
    if p.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(p)[1]


def _is_label_name(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in LABEL_HINTS)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an intensity array to uint8 grayscale."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def remap_labels(arr: np.ndarray, value_map: dict[int, int] | None) -> np.ndarray:
    """Apply an explicit {src_value: dst_value} remap; identity if None."""
    arr = np.asarray(arr)
    if not value_map:
        return arr.astype(np.uint8)
    out = np.zeros_like(arr, dtype=np.uint8)
    for src, dst in value_map.items():
        out[arr == src] = dst
    return out


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------
def inspect(data_root: str, keys: list[str]) -> None:
    for key in keys:
        base = raw_dir(data_root, key)
        print(f"\n=== {SOURCES[key]['folder']} : {base} ===")
        if key == "509":  # CAMUS is processed straight from CAMUS_public.zip
            zp = _find_camus_zip(data_root, key)
            print(f"  CAMUS_public.zip: {zp or 'NOT FOUND'}"
                  + ("" if zp else " — download it into the dataset folder"))
            continue
        if not os.path.isdir(base):
            print("  (no raw_source/ — run download_datasets.py first)")
            continue
        ext_counts: Counter = Counter()
        sample: dict[str, str] = {}
        total = 0
        for root, _dirs, files in os.walk(base):
            for f in files:
                e = _ext(f)
                ext_counts[e] += 1
                total += 1
                sample.setdefault(e, os.path.relpath(os.path.join(root, f), base))
        print(f"  files: {total}")
        for e, c in ext_counts.most_common():
            print(f"    {e or '<none>':10s} x{c:<7d} e.g. {sample[e]}")


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
def _save_png(arr: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


class Writer:
    """Accumulates image/label pairs into <Dataset>/{images,labels}Tr_generated."""

    def __init__(self, ds_dir: str, limit: int | None):
        self.img_dir = os.path.join(ds_dir, "imagesTr_generated")
        self.lab_dir = os.path.join(ds_dir, "labelsTr_generated")
        self.limit = limit
        self.records: list[dict] = []
        self._seen: set[str] = set()

    def _unique(self, stem: str) -> str:
        cand, i = stem, 1
        while cand in self._seen:
            cand, i = f"{stem}__{i}", i + 1
        self._seen.add(cand)
        return cand

    def add(self, stem: str, image: np.ndarray, label: np.ndarray | None) -> bool:
        if self.limit is not None and len(self.records) >= self.limit:
            return False
        stem = self._unique(stem)
        img_name = f"{stem}_0000.png"
        _save_png(to_uint8(image), os.path.join(self.img_dir, img_name))
        rec = {"image": f"imagesTr_generated/{img_name}"}
        if label is not None:
            lab_name = f"{stem}.png"
            _save_png(label.astype(np.uint8), os.path.join(self.lab_dir, lab_name))
            rec["label"] = f"labelsTr_generated/{lab_name}"
        self.records.append(rec)
        return True


# ---------------------------------------------------------------------------
# source readers
# ---------------------------------------------------------------------------
def _read_image_file(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def _read_label_file(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path))


def _iter_nifti_slices(path: str):
    """Yield (slice_index, 2D array) for a NIfTI volume (handles 2D/3D/4D)."""
    import nibabel as nib
    vol = np.asanyarray(nib.load(path).dataobj)
    vol = np.squeeze(vol)
    if vol.ndim == 2:
        yield 0, vol
    elif vol.ndim == 3:
        # assume last axis is the stack of frames/slices
        for i in range(vol.shape[-1]):
            yield i, vol[..., i]
    else:  # 4D+: flatten trailing dims
        flat = vol.reshape(vol.shape[0], vol.shape[1], -1)
        for i in range(flat.shape[-1]):
            yield i, flat[..., i]


def _iter_video_frames(path: str, stride: int = 1):
    import cv2
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield idx, gray
        idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# CAMUS (Dataset509) — faithful converter that reproduces the official filenames
# ---------------------------------------------------------------------------
def _find_camus_zip(data_root: str, key: str) -> str | None:
    """Locate CAMUS_public.zip in the dataset folder or its raw_source/."""
    for folder in (dataset_dir(data_root, key), raw_dir(data_root, key)):
        hits = glob.glob(os.path.join(folder, "**", "CAMUS_public.zip"), recursive=True)
        if hits:
            return hits[0]
    return None


def _load_nii_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Load a (possibly .gz) NIfTI volume straight from a zip member."""
    import nibabel as nib
    raw = zf.read(member)
    if member.endswith(".gz"):
        raw = gzip.decompress(raw)
    return np.asanyarray(nib.Nifti1Image.from_bytes(raw).dataobj)


_CAMUS_RE = re.compile(r"(patient\d+)_(\dCH)_t(\d+)_0000\.png$")


def convert_camus(data_root: str, key: str, limit: int | None) -> None:
    """Slice CAMUS half-sequence volumes into the EXACT official nnU-Net layout.

    Each `patientXXXX_{2CH,4CH}_half_sequence(.nii.gz)` + its `_gt` counterpart
    gives per-frame image+label (labels: 1=LV, 2=Myo, 3=LA, matching dataset.json).
    Frames listed in dataset.json's training set are written to imagesTr/labelsTr
    with their official names; patients absent from training (the held-out test
    split) are written in full to imagesTs/labelsTs.
    """
    from PIL import Image
    cfg = SOURCES[key]
    ds_dir = dataset_dir(data_root, key)
    _log(f"=== {cfg['folder']} (CAMUS faithful) ===")

    zpath = _find_camus_zip(data_root, key)
    if not zpath:
        _log("  CAMUS_public.zip not found in the dataset folder or raw_source/; skipping")
        return
    _log(f"  source: {zpath}")

    # 1) parse the official training set -> needed[(patient,view)] = {frame, ...}
    needed: dict[tuple[str, str], set[int]] = defaultdict(set)
    train_patients: set[str] = set()
    dj = os.path.join(ds_dir, "dataset.json")
    if os.path.isfile(dj):
        for rec in json.load(open(dj)).get("training", []):
            m = _CAMUS_RE.search(rec["image"])
            if m:
                needed[(m.group(1), m.group(2))].add(int(m.group(3)))
                train_patients.add(m.group(1))
    _log(f"  dataset.json training: {sum(len(v) for v in needed.values())} frames, "
         f"{len(train_patients)} patients")

    img_tr = os.path.join(ds_dir, "imagesTr")
    lab_tr = os.path.join(ds_dir, "labelsTr")
    img_ts = os.path.join(ds_dir, "imagesTs")
    lab_ts = os.path.join(ds_dir, "labelsTs")
    for d in (img_tr, lab_tr, img_ts, lab_ts):
        os.makedirs(d, exist_ok=True)

    def save(arr: np.ndarray, path: str, is_label: bool) -> None:
        if is_label:
            out = np.clip(arr, 0, 255).astype(np.uint8)
        else:  # CAMUS images are already 0..255 grayscale
            out = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(path)

    zf = zipfile.ZipFile(zpath)
    seqs = sorted(n for n in zf.namelist() if n.endswith("_half_sequence.nii.gz"))
    n_tr = n_ts = n_seq = 0
    for seq in seqs:
        m = re.search(r"(patient\d+)_(\dCH)_half_sequence\.nii\.gz$", seq)
        if not m:
            continue
        patient, view = m.group(1), m.group(2)
        gt = seq.replace("_half_sequence.nii.gz", "_half_sequence_gt.nii.gz")
        if gt not in zf.NameToInfo:
            _log(f"  WARNING: missing gt for {patient}_{view}; skipping")
            continue

        is_train = patient in train_patients
        vol = _load_nii_from_zip(zf, seq)
        lab = _load_nii_from_zip(zf, gt)
        n_frames = vol.shape[-1] if vol.ndim == 3 else 1
        frames = (sorted(needed[(patient, view)]) if is_train
                  else list(range(n_frames)))
        idir, ldir = (img_tr, lab_tr) if is_train else (img_ts, lab_ts)

        for t in frames:
            if t >= n_frames:
                _log(f"  WARNING: {patient}_{view} frame {t} >= {n_frames}; skipping")
                continue
            stem = f"{patient}_{view}_t{t}"
            save(vol[..., t] if vol.ndim == 3 else vol,
                 os.path.join(idir, f"{stem}_0000.png"), is_label=False)
            save(lab[..., t] if lab.ndim == 3 else lab,
                 os.path.join(ldir, f"{stem}.png"), is_label=True)
            if is_train:
                n_tr += 1
            else:
                n_ts += 1
        n_seq += 1
        if limit is not None and n_tr >= limit:
            _log(f"  (stopping early at --limit {limit})")
            break
        if n_seq % 200 == 0:
            _log(f"  ...{n_seq}/{len(seqs)} sequences, train={n_tr} test={n_ts}")

    # coverage check against dataset.json
    want = sum(len(v) for v in needed.values())
    have = sum(1 for v in needed for t in needed[v]
               if os.path.exists(os.path.join(img_tr, f"{v[0]}_{v[1]}_t{t}_0000.png")))
    _log(f"  wrote imagesTr/labelsTr={n_tr}  imagesTs/labelsTs={n_ts}")
    _log(f"  dataset.json coverage: {have}/{want} training frames present"
         + (" ✓ (dataset.json is now directly usable)" if have == want and want else ""))


# ---------------------------------------------------------------------------
# Four-Chamber (Dataset510) — faithful converter across CardiacUDC/CardiacNet/EchoCP
# (mapping rules reverse-engineered + verified to resolve 4503/4503 dataset.json refs)
# ---------------------------------------------------------------------------
_ECHO_RE = re.compile(r"^(\d+)_([vr])_(\d+)$")
_UDC_SITE_RE = re.compile(r"^(Site_[GR]_\d+)_(.+)_(\d+)$")
_UDC_LAF_RE = re.compile(r"^label_all_frame_(.+)_(\d+)$")
_NET_RE = re.compile(r"^(ASD|Non-ASD|PAH|Non-PAH)_(\d+)_(.+)_(\d+)$")
_NET_NOSTEM_RE = re.compile(r"^(ASD|Non-ASD|PAH|Non-PAH)_(\d+)_(\d+)$")


def _resolve_510(base: str, R: str):
    """Map a dataset.json basename -> (image_path, label_path|None, frame) or None.

    R is the raw_source/ directory. Paths returned are absolute.
    """
    # EchoCP: {id}_{v|r}_{frame}
    m = _ECHO_RE.match(base)
    if m:
        d = os.path.join(R, "EchoCP", "EchoCP_dataset")
        stem = f"{m.group(1)}_{m.group(2)}"
        return (os.path.join(d, f"{stem}_image.nii.gz"),
                os.path.join(d, f"{stem}_label.nii.gz"), int(m.group(3)))
    # CardiacUDC: label_all_frame_{stem}_{frame}
    m = _UDC_LAF_RE.match(base)
    if m:
        d = os.path.join(R, "CardiacUDC", "cardiacUDC_dataset", "label_all_frame")
        return (os.path.join(d, f"{m.group(1)}_image.nii.gz"),
                os.path.join(d, f"{m.group(1)}_label.nii.gz"), int(m.group(2)))
    # CardiacUDC: Site_X_NN_{stem}_{frame}
    m = _UDC_SITE_RE.match(base)
    if m:
        d = os.path.join(R, "CardiacUDC", "cardiacUDC_dataset", m.group(1))
        return (os.path.join(d, f"{m.group(2)}_image.nii.gz"),
                os.path.join(d, f"{m.group(2)}_label.nii.gz"), int(m.group(3)))
    # CardiacNet: {subgroup}_{id}_{stem}_{frame}  (image entry is usually a dir)
    m = _NET_RE.match(base) or _NET_NOSTEM_RE.match(base)
    if m:
        subgroup, idn = m.group(1), m.group(2)
        stem = m.group(3) if m.re is _NET_RE else None
        frame = int(m.group(m.re.groups))
        folder = "CardiacNet-ASD" if subgroup in ("ASD", "Non-ASD") else "CardiacNet-PAH"
        d = os.path.join(R, "CardiacNet", "CardiacNet", folder, subgroup)
        entry = os.path.join(d, f"{idn}_image.nii")
        if os.path.isdir(entry) and stem:
            img = os.path.join(entry, f"{stem}_image.nii")
        else:
            img = entry  # plain file
        return (img, os.path.join(d, f"{idn}_label.nii"), frame)
    return None


def _read_nii_frame_partial(path: str, frame: int) -> np.ndarray | None:
    """Read one frame from a possibly-truncated uncompressed NIfTI-1 (1-byte types).

    Some CardiacNet `_label.nii` files are truncated in the Kaggle upload, so
    nibabel refuses the whole volume; but the early frames are intact and can be
    read directly. Returns an (X, Y) array (matching nibabel's vol[:,:,frame]) or
    None if that frame's bytes are not present. Handles .nii only (not .gz)."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(352)
            if len(hdr) < 352:
                return None
            X = struct.unpack("<h", hdr[42:44])[0]
            Y = struct.unpack("<h", hdr[44:46])[0]
            datatype = struct.unpack("<h", hdr[70:72])[0]
            voxoff = int(struct.unpack("<f", hdr[108:112])[0])
            if datatype not in (2, 256):  # 2=uint8, 256=int8 (1 byte/voxel)
                return None
            f.seek(voxoff + frame * X * Y)
            buf = f.read(X * Y)
        if len(buf) < X * Y:
            return None
        dt = np.int8 if datatype == 256 else np.uint8
        return np.frombuffer(buf, dtype=dt).reshape((X, Y), order="F")
    except Exception:  # noqa: BLE001
        return None


def _img_u8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    mn, mx = float(a.min()), float(a.max())
    if mn >= 0 and mx <= 255:
        return a.astype(np.uint8)
    if mx <= mn:
        return np.zeros(a.shape, np.uint8)
    return ((a - mn) / (mx - mn) * 255).round().astype(np.uint8)


def _load_vol_safe(path: str):
    import nibabel as nib
    try:
        return np.squeeze(np.asanyarray(nib.load(path).dataobj))
    except Exception as exc:  # truncated/damaged volume
        _log(f"  WARNING: cannot read {os.path.basename(path)}: {exc}")
        return None


def _emit_510_groups(groups: dict, img_dir: str, lab_dir: str,
                     limit: int | None = None) -> tuple[int, int, int]:
    """Write {(img_path,lab_path): [(base,frame)]} as imagesXx/labelsXx PNGs.
    Returns (n_img, n_lab, n_missing_lab). Handles truncated label volumes via
    partial reads and remaps the lone CardiacUDC value 6 -> 4."""
    from PIL import Image
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    n_img = n_lab = n_missing = done = 0
    n_groups = len(groups)
    for gi, ((img_path, lab_path), items) in enumerate(groups.items(), 1):
        vol = _load_vol_safe(img_path)
        if vol is None:
            continue
        have_lab = bool(lab_path and os.path.exists(lab_path))
        lvol = _load_vol_safe(lab_path) if have_lab else None
        lab_partial = have_lab and lvol is None  # truncated -> per-frame partial read
        depth = vol.shape[-1] if vol.ndim == 3 else 1
        for base, frame in items:
            if frame >= depth:
                _log(f"  WARNING: {base} frame {frame} >= depth {depth}; skip")
                continue
            sl = vol[..., frame] if vol.ndim == 3 else vol
            Image.fromarray(_img_u8(sl)).save(os.path.join(img_dir, f"{base}_0000.png"))
            n_img += 1
            ls = None
            if lvol is not None and (lvol.ndim < 3 or frame < lvol.shape[-1]):
                ls = lvol[..., frame] if lvol.ndim == 3 else lvol
            elif lab_partial:
                ls = _read_nii_frame_partial(lab_path, frame)
            if ls is not None:
                ls = np.clip(np.where(ls == 6, 4, ls), 0, 4).astype(np.uint8)
                Image.fromarray(ls).save(os.path.join(lab_dir, f"{base}.png"))
                n_lab += 1
            else:
                n_missing += 1
            done += 1
            if limit is not None and done >= limit:
                break
        del vol, lvol
        if limit is not None and done >= limit:
            _log(f"  (stopping early at --limit {limit})")
            break
        if gi % 100 == 0:
            _log(f"  ...{gi}/{n_groups} volumes, {n_img} frames written")
    return n_img, n_lab, n_missing


def _nonempty_frames(path: str) -> list[int]:
    """Indices of label frames that contain any nonzero voxel (handles a
    truncated uncompressed .nii by scanning only its intact frames)."""
    import nibabel as nib
    try:
        v = np.squeeze(np.asanyarray(nib.load(path).dataobj))
        if v.ndim == 2:
            return [0] if v.any() else []
        nz = v.reshape(-1, v.shape[-1]).any(axis=0)
        return list(np.nonzero(nz)[0].tolist())
    except Exception:  # noqa: BLE001
        if not path.endswith(".nii"):
            return []
        frames, z = [], 0
        while True:
            fr = _read_nii_frame_partial(path, z)
            if fr is None:
                break
            if fr.any():
                frames.append(z)
            z += 1
        return frames


def _enumerate_510_labeled(R: str) -> dict:
    """All non-empty labeled frames across the 3 sources -> {base: (img,lab,frame)},
    with base names built by the same rule as the official dataset.json."""
    cands: dict[str, tuple[str, str, int]] = {}
    for lab in glob.glob(f"{R}/EchoCP/EchoCP_dataset/*_label.nii.gz"):
        stem = os.path.basename(lab)[: -len("_label.nii.gz")]
        img = lab.replace("_label.nii.gz", "_image.nii.gz")
        for fr in _nonempty_frames(lab):
            cands[f"{stem}_{fr:03d}"] = (img, lab, fr)
    for lab in glob.glob(f"{R}/CardiacUDC/cardiacUDC_dataset/*/*_label.nii.gz"):
        site = os.path.basename(os.path.dirname(lab))
        stem = os.path.basename(lab)[: -len("_label.nii.gz")]
        img = lab.replace("_label.nii.gz", "_image.nii.gz")
        pre = f"label_all_frame_{stem}" if site == "label_all_frame" else f"{site}_{stem}"
        for fr in _nonempty_frames(lab):
            cands[f"{pre}_{fr:03d}"] = (img, lab, fr)
    for lab in glob.glob(f"{R}/CardiacNet/CardiacNet/CardiacNet-*/*/*_label.nii"):
        subgroup = os.path.basename(os.path.dirname(lab))
        idn = os.path.basename(lab)[: -len("_label.nii")]
        entry = os.path.join(os.path.dirname(lab), f"{idn}_image.nii")
        if os.path.isdir(entry):
            inner = [x for x in os.listdir(entry) if x.endswith("_image.nii")]
            if not inner:
                continue
            img = os.path.join(entry, inner[0])
            pre = f"{subgroup}_{idn}_{inner[0][: -len('_image.nii')]}"
        elif os.path.isfile(entry):
            img, pre = entry, f"{subgroup}_{idn}"
        else:
            continue
        for fr in _nonempty_frames(lab):
            cands[f"{pre}_{fr:03d}"] = (img, lab, fr)
    return cands


def convert_four_chamber(data_root: str, key: str, limit: int | None,
                         with_test: bool = False) -> None:
    """Reproduce the official Dataset510 slices into imagesTr/labelsTr using the
    exact dataset.json names. Labels: identity, plus CardiacUDC value 6 -> 4.

    with_test: also build a leakage-free held-out test set (imagesTs/labelsTs) from
    every labeled frame whose source volume contributes NO training frame. This is
    a superset of the official 565 test split (which the public json doesn't list),
    but is guaranteed volume-disjoint from training."""
    ds_dir = dataset_dir(data_root, key)
    R = raw_dir(data_root, key)
    _log(f"=== {SOURCES[key]['folder']} (Four-Chamber faithful) ===")
    if not os.path.isdir(R):
        _log("  no raw_source/ — run download_datasets.py first; skipping")
        return

    dj = json.load(open(os.path.join(ds_dir, "dataset.json")))
    # resolve every training entry, then group by raw volume to load each once
    groups: dict[tuple[str, str | None], list[tuple[str, int]]] = defaultdict(list)
    unresolved = 0
    for rec in dj["training"]:
        base = os.path.basename(rec["image"])[: -len("_0000.png")]
        r = _resolve_510(base, R)
        if r is None or not os.path.exists(r[0]):
            unresolved += 1
            continue
        groups[(r[0], r[1])].append((base, r[2]))
    _log(f"  dataset.json training: {len(dj['training'])} | resolved volumes: "
         f"{len(groups)} | unresolved/missing: {unresolved}")

    n_img, n_lab, n_missing = _emit_510_groups(
        groups, os.path.join(ds_dir, "imagesTr"), os.path.join(ds_dir, "labelsTr"), limit)
    want = len(dj["training"])
    have = sum(1 for rec in dj["training"]
               if os.path.exists(os.path.join(ds_dir, rec["image"])))
    _log(f"  wrote imagesTr={n_img} labelsTr={n_lab} (labels missing for {n_missing})")
    _log(f"  dataset.json coverage: {have}/{want} training frames present"
         + (" ✓ (dataset.json is now directly usable)" if have == want and want else ""))

    if with_test and limit is None:
        _log("  building leakage-free held-out test set (imagesTs/labelsTs)...")
        train_vols = {lab for (_img, lab) in groups}
        clean = {b: v for b, v in _enumerate_510_labeled(R).items()
                 if v[1] not in train_vols}
        test_groups: dict = defaultdict(list)
        for b, (img, lab, fr) in clean.items():
            test_groups[(img, lab)].append((b, fr))
        ti, tl, _ = _emit_510_groups(
            test_groups, os.path.join(ds_dir, "imagesTs"), os.path.join(ds_dir, "labelsTs"))
        _log(f"  test set: imagesTs={ti} labelsTs={tl} "
             f"(from {len(test_groups)} fully held-out volumes; volume-disjoint from train)")


# ---------------------------------------------------------------------------
# generic auto-convert
# ---------------------------------------------------------------------------
def convert_dataset(data_root: str, key: str, limit: int | None,
                    video_stride: int) -> None:
    cfg = SOURCES[key]
    ds_dir = dataset_dir(data_root, key)
    base = raw_dir(data_root, key)
    _log(f"=== {cfg['folder']} ===")
    if not os.path.isdir(base):
        _log("  no raw_source/ — run download_datasets.py first; skipping")
        return

    writer = Writer(ds_dir, limit)

    # 1) index every file by extension
    images, labels, niftis, videos = [], [], [], []
    for root, _dirs, files in os.walk(base):
        for f in files:
            full = os.path.join(root, f)
            e = _ext(f)
            if e == ".nii.gz" or e == ".nii":
                niftis.append(full)
            elif e in VID_EXT:
                videos.append(full)
            elif e in IMG_EXT:
                (labels if _is_label_name(f) else images).append(full)

    # 2) pair plain images with masks that share a stem (best-effort)
    def stem_of(p: str) -> str:
        n = os.path.basename(p)
        if n.lower().endswith(".nii.gz"):
            n = n[:-7]
        else:
            n = os.path.splitext(n)[0]
        for h in LABEL_HINTS:
            n = n.replace(h, "").replace(h.upper(), "")
        return n.strip("_")

    label_by_stem: dict[str, str] = {}
    for lp in labels:
        label_by_stem.setdefault(stem_of(lp), lp)

    n_img = n_lab = n_nii = n_vid = 0

    # images (+ optional matching mask)
    for ip in sorted(images):
        st = stem_of(ip)
        lab = None
        if st in label_by_stem:
            lab = remap_labels(_read_label_file(label_by_stem[st]),
                               cfg.get("value_map"))
            n_lab += 1
        if not writer.add(f"{key}_img_{st}", _read_image_file(ip), lab):
            break
        n_img += 1

    # NIfTI volumes (+ matching *_gt volume sliced as labels)
    nii_labels = {stem_of(p): p for p in niftis if _is_label_name(os.path.basename(p))}
    for vp in sorted(p for p in niftis if not _is_label_name(os.path.basename(p))):
        st = stem_of(vp)
        lab_vol = nii_labels.get(st)
        lab_slices = dict(_iter_nifti_slices(lab_vol)) if lab_vol else {}
        for i, sl in _iter_nifti_slices(vp):
            lab = None
            if i in lab_slices:
                lab = remap_labels(lab_slices[i], cfg.get("value_map"))
            if not writer.add(f"{key}_{st}_t{i:03d}", sl, lab):
                break
        n_nii += 1

    # videos -> frames (no labels)
    for vp in sorted(videos):
        st = os.path.splitext(os.path.basename(vp))[0]
        for i, fr in _iter_video_frames(vp, stride=video_stride):
            if not writer.add(f"{key}_{st}_f{i:04d}", fr, None):
                break
        n_vid += 1

    # 3) write our manifest (never clobber the official dataset.json)
    manifest = {
        "name": cfg["folder"],
        "note": "Best-effort conversion from raw_source/. Filenames/counts "
                "do NOT match the official dataset.json (organizer preprocessing "
                "scripts are not public).",
        "channel_names": {"0": "Ultrasound"},
        "labels": cfg["labels"],
        "numTraining": len(writer.records),
        "file_ending": ".png",
        "training": writer.records,
    }
    out = os.path.join(ds_dir, "dataset_generated.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    with_lab = sum("label" in r for r in writer.records)
    _log(f"  images={n_img} (paired masks={n_lab}) niftis={n_nii} videos={n_vid}")
    _log(f"  wrote {len(writer.records)} samples ({with_lab} with labels) -> {out}")
    if with_lab == 0:
        _log("  WARNING: no segmentation labels were found/paired in the raw data; "
             "this dataset may only support detection (see dataset_detect.json).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--datasets", nargs="+", default=AVAILABLE_KEYS,
                    choices=AVAILABLE_KEYS,
                    help="Which datasets to convert (511 excluded: FUSEP removed from Kaggle)")
    ap.add_argument("--inspect", action="store_true",
                    help="Only print the raw_source/ file inventory, then exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max samples per dataset (for a quick test run)")
    ap.add_argument("--video-stride", type=int, default=1,
                    help="Keep every Nth video frame (default 1 = all)")
    ap.add_argument("--with-test", action="store_true",
                    help="Also build a leakage-free held-out test set "
                         "(imagesTs/labelsTs); currently used by Dataset510")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.data_root, args.datasets)
        return

    for key in args.datasets:
        try:
            if key == "509":  # CAMUS — faithful, official-naming converter
                convert_camus(args.data_root, key, args.limit)
            elif key == "510":  # Four-Chamber — faithful converter (3 sources)
                convert_four_chamber(args.data_root, key, args.limit,
                                     with_test=args.with_test)
            else:
                convert_dataset(args.data_root, key, args.limit, args.video_stride)
        except Exception as exc:  # noqa: BLE001
            _log(f"  ERROR converting {key}: {exc}")
    _log("Done.")


if __name__ == "__main__":
    main()
