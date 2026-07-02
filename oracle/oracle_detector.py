"""Oracle detector: simulates a detection tool by returning ground-truth boxes.

Drop-in stand-in for a real ultradetector. All scores are 1.0 and the source is
tagged so downstream code knows the boxes came from the annotation, not a model.
"""
from __future__ import annotations

from typing import Optional

from schemas import ORACLE_SOURCE


def detect(store, case_id, target: Optional[str] = None) -> dict:
    """Return ground-truth bounding boxes for ``target`` in ``case_id``.

    Shape::
        {"boxes": [[x_min, y_min, x_max, y_max], ...],
         "labels": [...], "scores": [1.0, ...],
         "source": "ground_truth_annotation"}
    """
    objs = store.get_objects(case_id, target)
    boxes, labels = [], []
    for o in objs:
        if o.bbox is None:
            continue
        boxes.append(list(o.bbox))
        labels.append(o.name)
    return {
        "boxes": boxes,
        "labels": labels,
        "scores": [1.0] * len(boxes),
        "source": ORACLE_SOURCE,
    }
