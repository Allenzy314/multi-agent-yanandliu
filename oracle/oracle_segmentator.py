"""Oracle segmentator: simulates a segmentation tool by returning ground-truth
mask paths (relative to the annotation store). Stand-in for a real
ultrasegmentator.
"""
from __future__ import annotations

from typing import Optional

from schemas import ORACLE_SOURCE


def segment(store, case_id, target: Optional[str] = None) -> dict:
    """Return ground-truth mask paths for ``target`` in ``case_id``.

    Shape::
        {"mask_paths": [...], "labels": [...],
         "source": "ground_truth_annotation"}

    Paths are returned as stored (relative); callers resolve them via
    ``store.resolve_path``.
    """
    objs = store.get_objects(case_id, target)
    mask_paths, labels = [], []
    for o in objs:
        if o.mask_path is None:
            continue
        mask_paths.append(o.mask_path)
        labels.append(o.name)
    return {"mask_paths": mask_paths, "labels": labels, "source": ORACLE_SOURCE}
