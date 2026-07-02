"""Convert a FLARE25 nnU-Net-style ultrasound dataset into the unified
annotation format consumed by :class:`oracle.annotation_store.AnnotationStore`.

A single ``DatasetXXX_*`` folder (under ``flare_dataset/Ultrasound/``) may carry
up to three task descriptions that we merge *per image*:

``dataset.json`` (segmentation, nnU-Net)
    ``labels`` maps name -> class int (``{"background": 0, "Thyroid_Nodule": 1}``);
    ``training`` is a list of ``{"image": "imagesTr/0000_0000.png",
    "label": "labelsTr/0000.png"}``. The label PNG is class-indexed: pixel value
    equals the class int.

``dataset_detect.json`` (detection)
    ``labels`` maps name -> 0-based class int (``{"Thyroid_Nodule": 0}``);
    ``training`` is a list of ``{"image": "imagesTr/0415.png",
    "boxes": [{"class": 0, "points": [[x, y], ...]}]}``. Points are *normalized*
    ``[0, 1]`` axis-aligned box corners; denormalize with the image pixel size.

``dataset_cls.json`` (classification, only some datasets)
    ``labels`` maps name -> int (``{"benign": 0, "Malignant": 1}``);
    ``training`` is a list of ``{"image": "imagesTr/0000_0000.png", "label": 0}``.

Merge key
    Records across the three files are joined by the *stem key*: the image
    filename with its extension removed, truncated at the first underscore. This
    collapses the nnU-Net channel suffix so ``0000_0000.png`` (seg / cls) and
    ``0000.png`` (detect / label) both map to ``0000``. The stem key is also the
    emitted ``case_id``.

The unified output is a JSON list of records (see ``AnnotationStore`` docstring).
Image and mask paths are written *relative to the output file's directory* so the
store resolves them without an extra ``--image_dir``.

This is a best-effort utility meant to run against real data; it fails gracefully
(warnings, not crashes) on missing files and never requires the dataset to be
present at import time.

Usage::

    python scripts/build_unified_annotations.py \
        --dataset_dir flare_dataset/Ultrasound/Dataset506_Thyroid_Nodule \
        --out /tmp/thyroid/annotation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _stem_key(path: str) -> str:
    """Merge key for an image/label path.

    ``imagesTr/0000_0000.png`` -> ``0000``; ``imagesTr/0415.png`` -> ``0415``.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_", 1)[0]


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, ValueError) as exc:
        _warn(f"could not read {path}: {exc}")
        return None
    if not isinstance(obj, dict):
        _warn(f"{path}: expected a JSON object, got {type(obj).__name__}; ignoring")
        return None
    return obj


def _invert_labels(labels: Optional[dict]) -> dict[int, str]:
    """``{name: int}`` -> ``{int: name}`` (ignores non-int values)."""
    out: dict[int, str] = {}
    if not isinstance(labels, dict):
        return out
    for name, val in labels.items():
        try:
            out[int(val)] = name
        except (TypeError, ValueError):
            continue
    return out


def _entry_image(entry: Any) -> Optional[str]:
    """Extract the image path from a training/test entry (dict or bare str)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("image")
    return None


def _relpath(target: str, start_dir: str) -> str:
    """Path to ``target`` relative to ``start_dir`` (POSIX-style separators)."""
    return os.path.relpath(target, start_dir).replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def _index_image_files(images_dir: str) -> dict[str, str]:
    """Map stem key -> absolute image path for every file in ``images_dir``."""
    index: dict[str, str] = {}
    if not os.path.isdir(images_dir):
        return index
    for name in sorted(os.listdir(images_dir)):
        full = os.path.join(images_dir, name)
        if os.path.isfile(full):
            # first file for a key wins; sorted() makes this deterministic
            index.setdefault(_stem_key(name), full)
    return index


def _denorm_box(points: list, width: int, height: int) -> Optional[list]:
    """Normalized polygon corners -> pixel ``[x_min, y_min, x_max, y_max]``."""
    xs, ys = [], []
    for pt in points or []:
        try:
            xs.append(float(pt[0]) * width)
            ys.append(float(pt[1]) * height)
        except (TypeError, ValueError, IndexError):
            return None
    if not xs or not ys:
        return None
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _mask_channels(
    mask_arr: "np.ndarray",
    seg_names: dict[int, str],
    single_structure: bool,
) -> list[tuple[str, "np.ndarray"]]:
    """Split a label array into ``(class_name, binary_uint8_mask)`` channels.

    Clean nnU-Net masks are class-indexed (pixel value == class int). Some
    single-class datasets ship *antialiased* binary masks instead (background 0,
    foreground near 255 with soft edges), so a raw value split would emit dozens
    of spurious ``class_<v>`` channels. Detection: if the nonzero values are not
    all known class ints and the dataset is single-structure, treat the whole
    nonzero region as the one foreground class.
    """
    fg_ids = [i for i in seg_names if i != 0]
    present = [int(v) for v in np.unique(mask_arr) if int(v) != 0]
    if not present:
        return []

    unexpected = [v for v in present if v not in seg_names]
    if unexpected and single_structure and fg_ids:
        cls_name = seg_names.get(fg_ids[0], f"class_{fg_ids[0]}")
        binary = (np.asarray(mask_arr) > 0).astype(np.uint8) * 255
        return [(cls_name, binary)]

    channels: list[tuple[str, "np.ndarray"]] = []
    for val in present:
        if val not in seg_names:
            # unknown value in a multiclass mask: skip rather than invent a class
            _warn(f"label value {val} not in labels map; skipping that region")
            continue
        cls_name = seg_names[val]
        binary = (np.asarray(mask_arr) == val).astype(np.uint8) * 255
        channels.append((cls_name, binary))
    return channels


def build(
    dataset_dir: str,
    out_path: str,
    split: str = "Tr",
    masks_dir: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> str:
    """Convert ``dataset_dir`` and write the unified annotation JSON to ``out_path``.

    Returns ``out_path``. Raises ``FileNotFoundError`` only if the dataset folder
    itself is missing; per-case IO problems are downgraded to warnings.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    split = split.strip()
    images_sub = f"images{split}"
    labels_sub = f"labels{split}"
    images_dir = os.path.join(dataset_dir, images_sub)
    labels_dir = os.path.join(dataset_dir, labels_sub)
    list_key = "training" if split == "Tr" else "test"

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if masks_dir is None:
        masks_dir = os.path.join(out_dir, "masks")

    ds_seg = _load_json(os.path.join(dataset_dir, "dataset.json"))
    ds_det = _load_json(os.path.join(dataset_dir, "dataset_detect.json"))
    ds_cls = _load_json(os.path.join(dataset_dir, "dataset_cls.json"))
    if ds_seg is None and ds_det is None and ds_cls is None:
        raise FileNotFoundError(
            f"no dataset.json / dataset_detect.json / dataset_cls.json in {dataset_dir}"
        )

    # dataset name for records
    dataset_name = None
    for d in (ds_seg, ds_det, ds_cls):
        if d and d.get("name"):
            dataset_name = str(d["name"]).lower()
            break
    if not dataset_name:
        dataset_name = os.path.basename(os.path.abspath(dataset_dir.rstrip("/"))).lower()

    seg_names = _invert_labels(ds_seg.get("labels")) if ds_seg else {}
    det_names = _invert_labels(ds_det.get("labels")) if ds_det else {}
    cls_names = _invert_labels(ds_cls.get("labels")) if ds_cls else {}
    # "single-structure" = exactly one foreground segmentation/detection class
    fg_seg = [v for k, v in seg_names.items() if k != 0]
    single_structure = len(det_names) == 1 or len(fg_seg) == 1

    img_index = _index_image_files(images_dir)
    if not img_index:
        _warn(f"no image files found under {images_dir}")

    # ------------------------------------------------------------------ #
    # 1. discover the ordered set of stem keys across all task lists
    # ------------------------------------------------------------------ #
    order: list[str] = []
    seen: set[str] = set()

    def _register(entries: Any) -> None:
        for entry in entries or []:
            img = _entry_image(entry)
            if not img:
                continue
            key = _stem_key(img)
            if key not in seen:
                seen.add(key)
                order.append(key)

    for d in (ds_det, ds_seg, ds_cls):
        if d:
            _register(d.get(list_key))

    # ------------------------------------------------------------------ #
    # 2. lookups keyed by stem
    # ------------------------------------------------------------------ #
    det_by_key: dict[str, list] = {}
    if ds_det:
        for entry in ds_det.get(list_key) or []:
            img = _entry_image(entry)
            if img and isinstance(entry, dict):
                det_by_key[_stem_key(img)] = entry.get("boxes") or []

    seg_label_by_key: dict[str, str] = {}
    if ds_seg:
        for entry in ds_seg.get(list_key) or []:
            if isinstance(entry, dict) and entry.get("label"):
                img = _entry_image(entry)
                if img:
                    seg_label_by_key[_stem_key(img)] = entry["label"]

    cls_by_key: dict[str, int] = {}
    if ds_cls:
        for entry in ds_cls.get(list_key) or []:
            if isinstance(entry, dict) and entry.get("label") is not None:
                img = _entry_image(entry)
                if img:
                    try:
                        cls_by_key[_stem_key(img)] = int(entry["label"])
                    except (TypeError, ValueError):
                        continue

    # ------------------------------------------------------------------ #
    # 3. per-case conversion
    # ------------------------------------------------------------------ #
    records: list[dict] = []
    n_boxes = n_masks = n_labels = n_skipped = 0

    for key in order:
        if max_cases is not None and len(records) >= max_cases:
            break

        img_path = img_index.get(key)
        if not img_path:
            _warn(f"{key}: no image file under {images_sub}; skipping case")
            n_skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except (OSError, ValueError) as exc:
            _warn(f"{key}: cannot open image {img_path}: {exc}; skipping case")
            n_skipped += 1
            continue

        objects: list[dict] = []

        # --- detection boxes ------------------------------------------ #
        case_has_box = False
        for box in det_by_key.get(key, []):
            if not isinstance(box, dict):
                continue
            bbox = _denorm_box(box.get("points"), width, height)
            if bbox is None:
                continue
            cls_int = box.get("class")
            try:
                name = det_names.get(int(cls_int))
            except (TypeError, ValueError):
                name = None
            objects.append(
                {
                    "name": name,
                    "class_label": None,
                    "bbox": bbox,
                    "mask_path": None,
                    "contour": None,
                }
            )
            case_has_box = True
        if case_has_box:
            n_boxes += 1

        # --- segmentation masks --------------------------------------- #
        case_has_mask = False
        label_rel = seg_label_by_key.get(key)
        if label_rel:
            label_path = os.path.join(dataset_dir, label_rel)
            if os.path.isfile(label_path):
                try:
                    mask_arr = np.asarray(Image.open(label_path))
                except (OSError, ValueError) as exc:
                    _warn(f"{key}: cannot open label {label_path}: {exc}")
                    mask_arr = None
                if mask_arr is not None:
                    os.makedirs(masks_dir, exist_ok=True)
                    for cls_name, binary in _mask_channels(
                        mask_arr, seg_names, single_structure
                    ):
                        mask_fname = f"{key}_{cls_name}.png"
                        mask_abs = os.path.join(masks_dir, mask_fname)
                        try:
                            Image.fromarray(binary, mode="L").save(mask_abs)
                        except (OSError, ValueError) as exc:
                            _warn(f"{key}: cannot write mask {mask_abs}: {exc}")
                            continue
                        mask_rel = _relpath(mask_abs, out_dir)
                        # attach to a same-named box object lacking a mask,
                        # else create a new object for this class
                        attached = False
                        for obj in objects:
                            if obj["name"] == cls_name and obj["mask_path"] is None:
                                obj["mask_path"] = mask_rel
                                attached = True
                                break
                        if not attached:
                            objects.append(
                                {
                                    "name": cls_name,
                                    "class_label": None,
                                    "bbox": None,
                                    "mask_path": mask_rel,
                                    "contour": None,
                                }
                            )
                        case_has_mask = True
            else:
                _warn(f"{key}: label file missing: {label_path}")
        if case_has_mask:
            n_masks += 1

        # --- classification ------------------------------------------- #
        image_label = None
        if key in cls_by_key:
            image_label = cls_names.get(cls_by_key[key], str(cls_by_key[key]))
            n_labels += 1
            if single_structure and objects:
                objects[0]["class_label"] = image_label

        records.append(
            {
                "case_id": key,
                "dataset": dataset_name,
                "modality": "ultrasound",
                "image_path": _relpath(img_path, out_dir),
                "image_size": [width, height],
                "spacing": None,
                "objects": objects,
                "image_label": image_label,
                "measurements": {},
            }
        )

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(
        f"wrote {out_path}: {len(records)} cases "
        f"({n_boxes} with boxes, {n_masks} with masks, {n_labels} with labels, "
        f"{n_skipped} skipped)"
    )
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a FLARE25 nnU-Net ultrasound dataset to the unified "
        "annotation format.",
    )
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="a DatasetXXX_* folder under flare_dataset/Ultrasound/",
    )
    ap.add_argument(
        "--split",
        default="Tr",
        help="'Tr' (training lists) or 'Ts' (test lists); default Tr",
    )
    ap.add_argument("--out", required=True, help="path to write the unified .json")
    ap.add_argument(
        "--masks_dir",
        default=None,
        help="dir for per-object binary masks (default <out_dir>/masks)",
    )
    ap.add_argument(
        "--image_root",
        default=None,
        help="unused placeholder; images are referenced relative to --out",
    )
    ap.add_argument("--max_cases", type=int, default=None, help="limit number of cases")
    args = ap.parse_args(argv)

    try:
        build(
            dataset_dir=args.dataset_dir,
            out_path=args.out,
            split=args.split,
            masks_dir=args.masks_dir,
            max_cases=args.max_cases,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
