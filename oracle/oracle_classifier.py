"""Oracle classifier: simulates a classification tool by returning the
ground-truth class label. Stand-in for a real ultraclassifier.

Prefers a matching object's ``class_label``; falls back to the image-level
label when the target has no per-object class.
"""
from __future__ import annotations

from typing import Optional

from schemas import ORACLE_SOURCE


def classify(store, case_id, target: Optional[str] = None) -> dict:
    """Return the ground-truth class label for ``target`` in ``case_id``.

    Shape::
        {"label": "...", "probs": {"...": 1.0},
         "source": "ground_truth_annotation"}
    """
    label = None
    for o in store.get_objects(case_id, target):
        if o.class_label is not None:
            label = o.class_label
            break
    if label is None:
        label = store.get_image_label(case_id)
    probs = {label: 1.0} if label is not None else {}
    return {"label": label, "probs": probs, "source": ORACLE_SOURCE}
