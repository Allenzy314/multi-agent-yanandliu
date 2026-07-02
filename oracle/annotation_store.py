"""Unified annotation store.

Loads a single unified-annotation JSON file and exposes ground-truth lookups by
``case_id`` / ``target``. This is the *only* component that reads the full
annotation; the LLM never touches it.

Unified record format::

    {
      "case_id": "case_000001",
      "dataset": "thyroid_nodule",
      "modality": "ultrasound",
      "image_path": "images/case_000001.png",
      "image_size": [512, 512],           # [width, height]
      "spacing": null,                     # null or [sx, sy] mm/pixel
      "objects": [
        {"name": "thyroid_nodule", "class_label": "benign",
         "bbox": [x_min, y_min, x_max, y_max], "mask_path": "masks/...png",
         "contour": null}
      ],
      "image_label": "benign",
      "measurements": {}
    }

The file itself may be a list of records, ``{"cases": [...]}``, or a dict keyed
by ``case_id`` — all are normalized on load.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


@dataclass
class OracleObject:
    name: Optional[str] = None
    class_label: Optional[str] = None
    bbox: Optional[list] = None  # [x_min, y_min, x_max, y_max] in pixels
    mask_path: Optional[str] = None
    contour: Optional[Any] = None


@dataclass
class Case:
    case_id: str
    dataset: Optional[str] = None
    modality: Optional[str] = None
    image_path: Optional[str] = None
    image_size: Optional[list] = None  # [width, height]
    spacing: Optional[list] = None  # None or [sx, sy]
    objects: List[OracleObject] = field(default_factory=list)
    image_label: Optional[str] = None
    measurements: dict = field(default_factory=dict)


def _normalize_records(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "annotations", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        # dict keyed by case_id
        recs = []
        for k, v in data.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("case_id", k)
                recs.append(v)
        if recs:
            return recs
    raise ValueError("unrecognized annotation file structure")


def _record_to_case(r: dict) -> Case:
    objs = []
    for o in r.get("objects") or []:
        objs.append(
            OracleObject(
                name=o.get("name"),
                class_label=o.get("class_label"),
                bbox=o.get("bbox"),
                mask_path=o.get("mask_path"),
                contour=o.get("contour"),
            )
        )
    return Case(
        case_id=str(r.get("case_id")),
        dataset=r.get("dataset"),
        modality=r.get("modality"),
        image_path=r.get("image_path"),
        image_size=r.get("image_size"),
        spacing=r.get("spacing"),
        objects=objs,
        image_label=r.get("image_label"),
        measurements=r.get("measurements") or {},
    )


class AnnotationStore:
    def __init__(self, cases, base_dir: Optional[str] = None, image_dir: Optional[str] = None):
        self._cases = {c.case_id: c for c in cases}
        self._order = [c.case_id for c in cases]
        self.base_dir = base_dir
        self.image_dir = image_dir

    @classmethod
    def from_file(cls, annotation_file: str, image_dir: Optional[str] = None) -> "AnnotationStore":
        with open(annotation_file) as f:
            data = json.load(f)
        cases = [_record_to_case(r) for r in _normalize_records(data)]
        base_dir = os.path.dirname(os.path.abspath(annotation_file))
        return cls(cases, base_dir=base_dir, image_dir=image_dir)

    # --- introspection -----------------------------------------------------
    def case_ids(self) -> list:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, case_id) -> bool:
        return case_id in self._cases

    def get_case(self, case_id) -> Case:
        if case_id not in self._cases:
            raise KeyError(f"unknown case_id: {case_id}")
        return self._cases[case_id]

    def cases(self) -> list:
        return [self._cases[c] for c in self._order]

    # --- ground-truth lookups ---------------------------------------------
    def get_objects(self, case_id, target: Optional[str] = None) -> List[OracleObject]:
        """Objects for a case. If ``target`` is given, only objects whose name
        matches (case-insensitive) are returned."""
        objs = self.get_case(case_id).objects
        if target is None:
            return list(objs)
        t = _norm(target)
        return [o for o in objs if _norm(o.name) == t]

    def get_image_label(self, case_id) -> Optional[str]:
        return self.get_case(case_id).image_label

    def get_measurements(self, case_id) -> dict:
        return dict(self.get_case(case_id).measurements)

    def targets(self, case_id) -> list:
        """Distinct object names present in a case (order-preserving)."""
        seen, out = set(), []
        for o in self.get_case(case_id).objects:
            if o.name and o.name not in seen:
                seen.add(o.name)
                out.append(o.name)
        return out

    # --- path resolution ---------------------------------------------------
    def resolve_path(self, rel: Optional[str]) -> Optional[str]:
        """Resolve a mask/image path relative to ``image_dir`` then ``base_dir``."""
        if rel is None:
            return None
        if os.path.isabs(rel):
            return rel
        for root in (self.image_dir, self.base_dir):
            if root:
                p = os.path.join(root, rel)
                if os.path.exists(p):
                    return p
        root = self.image_dir or self.base_dir or "."
        return os.path.join(root, rel)
