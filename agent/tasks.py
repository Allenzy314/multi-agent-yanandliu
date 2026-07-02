"""Task registry: question templates, the canonical operation sequence, and the
deterministic ground-truth computation used for evaluation.

Ground truth is computed with the *same* measurement functions the executor
uses, so the oracle baseline should score near-perfect — any deviation points to
a real bug rather than tool error.
"""
from __future__ import annotations

from dataclasses import dataclass

from postprocess import measurement as M
from schemas import (
    ANSWER_BBOX,
    ANSWER_COUNT,
    ANSWER_LABEL,
    ANSWER_NUMBER,
    OP_CLASSIFY,
    OP_DETECT,
    OP_MEASURE_AREA,
    OP_MEASURE_COUNT,
    OP_MEASURE_MAX_DIAMETER,
    OP_SEGMENT,
    UNIT_PIXEL,
)


def _norm(s) -> str:
    return (s or "").strip().lower()


def _pretty(target) -> str:
    return (target or "structure").replace("_", " ")


@dataclass
class Task:
    name: str
    question_template: str  # uses {target}
    expected_ops: tuple
    answer_type: str
    unit: str = UNIT_PIXEL

    def question_for(self, target) -> str:
        return self.question_template.format(target=_pretty(target))


TASKS = {
    "detect": Task(
        "detect",
        "Detect the {target} in this ultrasound image and report its bounding "
        "box as [x_min, y_min, x_max, y_max] in pixel coordinates.",
        (OP_DETECT,),
        ANSWER_BBOX,
    ),
    "classify": Task(
        "classify",
        "Classify the {target} shown in this ultrasound image and report the "
        "class label.",
        (OP_CLASSIFY,),
        ANSWER_LABEL,
    ),
    "count": Task(
        "count",
        "How many {target} instances are present in this ultrasound image?",
        (OP_DETECT, OP_MEASURE_COUNT),
        ANSWER_COUNT,
    ),
    "measure_area": Task(
        "measure_area",
        "What is the area of the {target} in this ultrasound image? Report the "
        "value in pixels.",
        (OP_SEGMENT, OP_MEASURE_AREA),
        ANSWER_NUMBER,
    ),
    "measure_max_diameter": Task(
        "measure_max_diameter",
        "What is the maximum diameter of the {target} in this ultrasound image? "
        "Report the value in pixels.",
        (OP_DETECT, OP_MEASURE_MAX_DIAMETER),
        ANSWER_NUMBER,
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choices: {sorted(TASKS)}")
    return TASKS[name]


def _usable(task: Task, o) -> bool:
    """Whether an object carries what this task needs."""
    if task.name in ("detect", "count", "measure_max_diameter"):
        return o.bbox is not None
    if task.name == "measure_area":
        return o.mask_path is not None
    return True  # classify (uses class_label or image_label)


def objects_for_task(task: Task, case) -> list:
    return [o for o in case.objects if _usable(task, o)]


def target_for_task(task: Task, case):
    """The structure name this task asks about (first usable object, else the
    first object)."""
    objs = objects_for_task(task, case)
    if objs:
        return objs[0].name
    if case.objects:
        return case.objects[0].name
    return None


def applicable(task: Task, case) -> bool:
    """Whether this case can be evaluated for this task."""
    if task.name == "classify":
        if case.image_label is not None and case.objects:
            return True
        return any(o.class_label is not None for o in case.objects)
    return len(objects_for_task(task, case)) > 0


def ground_truth(task: Task, case, store):
    """Reference answer for ``case`` under ``task``.

    Returns ``{"answer_type", "value", "unit"}`` or None when not applicable.
    ``store`` is needed only to load masks (measure_area).
    """
    target = target_for_task(task, case)
    t = _norm(target)
    objs = objects_for_task(task, case)

    if task.name == "detect":
        for o in objs:
            if _norm(o.name) == t and o.bbox is not None:
                return {"answer_type": ANSWER_BBOX, "value": list(o.bbox), "unit": None}
        return None

    if task.name == "count":
        n = sum(1 for o in objs if _norm(o.name) == t)
        return {"answer_type": ANSWER_COUNT, "value": int(n), "unit": None}

    if task.name == "measure_max_diameter":
        # "maximum diameter" of the structure = its longest caliper. From a
        # bounding box the closest correct proxy is the longer side (the major
        # axis of the inscribed region), not the box diagonal, which would
        # overstate it by up to sqrt(2).
        vals = [
            M.max_diameter_from_bbox(o.bbox, mode="max_side")
            for o in objs
            if _norm(o.name) == t and o.bbox is not None
        ]
        if not vals:
            return None
        v = M.to_units(max(vals), case.spacing, "length", task.unit)
        unit = task.unit if case.spacing else UNIT_PIXEL
        return {"answer_type": ANSWER_NUMBER, "value": float(v), "unit": unit}

    if task.name == "measure_area":
        total, found = 0, False
        for o in objs:
            if _norm(o.name) != t or not o.mask_path:
                continue
            try:
                mask = M.load_mask(store.resolve_path(o.mask_path))
            except Exception:
                continue
            total += M.mask_area(mask)
            found = True
        if not found:
            return None
        v = M.to_units(total, case.spacing, "area", task.unit)
        unit = task.unit if case.spacing else UNIT_PIXEL
        return {"answer_type": ANSWER_NUMBER, "value": float(v), "unit": unit}

    if task.name == "classify":
        for o in objs:
            if _norm(o.name) == t and o.class_label is not None:
                return {"answer_type": ANSWER_LABEL, "value": o.class_label, "unit": None}
        # fall back to any object's class label, then the image label
        for o in case.objects:
            if o.class_label is not None:
                return {"answer_type": ANSWER_LABEL, "value": o.class_label, "unit": None}
        if case.image_label is not None:
            return {"answer_type": ANSWER_LABEL, "value": case.image_label, "unit": None}
        return None

    return None
