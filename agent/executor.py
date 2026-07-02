"""Executor: runs a JSON workflow plan against the oracle backends and derives a
final answer through deterministic post-processing.

The executor is the only place operations are mapped to oracle calls + numeric
post-processing. It records everything needed for a full audit log: each step's
oracle output and errors, the collected oracle outputs, and the final answer.
"""
from __future__ import annotations

from oracle import oracle_classifier, oracle_detector, oracle_segmentator
from postprocess import measurement as M
from schemas import (
    ANSWER_BBOX,
    ANSWER_COUNT,
    ANSWER_LABEL,
    ANSWER_NUMBER,
    OP_CLASSIFY,
    OP_DETECT,
    OP_MEASURE_AREA,
    OP_MEASURE_CIRCUMFERENCE,
    OP_MEASURE_COUNT,
    OP_MEASURE_MAX_DIAMETER,
    OP_SEGMENT,
    UNIT_PIXEL,
)


def _spacing(store, case_id):
    try:
        return store.get_case(case_id).spacing
    except Exception:
        return None


def _load_masks_area(store, case_id, paths):
    total, found = 0, False
    for p in paths or []:
        try:
            mask = M.load_mask(store.resolve_path(p))
        except Exception:
            continue
        total += M.mask_area(mask)
        found = True
    return total, found


def _run_op(op, target, unit, store, case_id, state):
    """Execute a single operation, mutating ``state`` and returning its output."""
    if op == OP_DETECT:
        out = oracle_detector.detect(store, case_id, target)
        state["boxes"] = out.get("boxes")
        return out
    if op == OP_SEGMENT:
        out = oracle_segmentator.segment(store, case_id, target)
        state["mask_paths"] = out.get("mask_paths")
        return out
    if op == OP_CLASSIFY:
        out = oracle_classifier.classify(store, case_id, target)
        state["label"] = out.get("label")
        return out

    if op == OP_MEASURE_COUNT:
        boxes = state.get("boxes")
        if boxes is None:
            boxes = oracle_detector.detect(store, case_id, target).get("boxes")
        return {"value": M.count_objects(boxes or []), "unit": "count"}

    if op == OP_MEASURE_MAX_DIAMETER:
        boxes = state.get("boxes")
        if boxes is None:
            boxes = oracle_detector.detect(store, case_id, target).get("boxes")
        if not boxes:
            return {"value": None, "unit": unit}
        # longer bbox side (major axis) — see agent/tasks.py measure_max_diameter
        v = max(M.max_diameter_from_bbox(b, mode="max_side") for b in boxes)
        v = M.to_units(v, _spacing(store, case_id), "length", unit)
        return {"value": float(v), "unit": unit}

    if op == OP_MEASURE_AREA:
        paths = state.get("mask_paths")
        if paths is None:
            paths = oracle_segmentator.segment(store, case_id, target).get("mask_paths")
        total, found = _load_masks_area(store, case_id, paths)
        if not found:
            return {"value": None, "unit": unit}
        v = M.to_units(total, _spacing(store, case_id), "area", unit)
        return {"value": float(v), "unit": unit}

    if op == OP_MEASURE_CIRCUMFERENCE:
        paths = state.get("mask_paths")
        if paths is None:
            paths = oracle_segmentator.segment(store, case_id, target).get("mask_paths")
        total, found = 0.0, False
        for p in paths or []:
            try:
                mask = M.load_mask(store.resolve_path(p))
            except Exception:
                continue
            total += M.contour_perimeter(mask)
            found = True
        if found:
            v = M.to_units(total, _spacing(store, case_id), "length", unit)
            return {"value": float(v), "unit": unit}
        # fallback: perimeter of the first detected box
        boxes = state.get("boxes")
        if boxes is None:
            boxes = oracle_detector.detect(store, case_id, target).get("boxes")
        if boxes:
            v = M.to_units(M.bbox_perimeter(boxes[0]), _spacing(store, case_id), "length", unit)
            return {"value": float(v), "unit": unit}
        return {"value": None, "unit": unit}

    raise ValueError(f"unknown operation {op!r}")


def _answer_from_step(rec):
    op = rec["op"]
    out = rec.get("oracle_output") or {}
    unit = rec.get("unit")
    if op == OP_DETECT:
        boxes = out.get("boxes") or []
        return {"answer_type": ANSWER_BBOX, "value": (list(boxes[0]) if boxes else None), "unit": None}
    if op == OP_CLASSIFY:
        return {"answer_type": ANSWER_LABEL, "value": out.get("label"), "unit": None}
    if op == OP_MEASURE_COUNT:
        return {"answer_type": ANSWER_COUNT, "value": out.get("value"), "unit": None}
    if op in (OP_MEASURE_AREA, OP_MEASURE_MAX_DIAMETER, OP_MEASURE_CIRCUMFERENCE):
        return {"answer_type": ANSWER_NUMBER, "value": out.get("value"), "unit": out.get("unit", unit)}
    if op == OP_SEGMENT:
        return {"answer_type": ANSWER_COUNT, "value": len(out.get("mask_paths") or []), "unit": None}
    return None


def _final_answer(answer_from, steps_out):
    if not steps_out:
        return None
    chosen = None
    if answer_from is not None:
        for rec in reversed(steps_out):
            if rec["op"] == answer_from and rec.get("error") is None:
                chosen = rec
                break
        if chosen is None:
            try:
                idx = int(answer_from)
                if 0 <= idx < len(steps_out):
                    chosen = steps_out[idx]
            except (ValueError, TypeError):
                pass
    if chosen is None:
        for rec in reversed(steps_out):
            if rec.get("error") is None:
                chosen = rec
                break
    if chosen is None:
        return None
    return _answer_from_step(chosen)


def execute_plan(plan_dict, store, case_id) -> dict:
    """Run a plan dict against the oracle backends.

    Returns::
        {"steps": [...], "oracle_outputs": {op: out}, "final_answer": {...}|None,
         "errors": [...]}
    """
    if not plan_dict or not isinstance(plan_dict, dict):
        return {"steps": [], "oracle_outputs": {}, "final_answer": None, "errors": ["missing or invalid plan"]}

    steps = plan_dict.get("steps") or []
    answer_from = plan_dict.get("answer_from")
    steps_out, oracle_outputs, errors = [], {}, []
    state = {"boxes": None, "mask_paths": None, "label": None}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {i}: not an object")
            continue
        op = step.get("op")
        target = step.get("target")
        unit = step.get("unit") or UNIT_PIXEL
        rec = {"index": i, "op": op, "target": target, "unit": unit, "oracle_output": None, "error": None}
        try:
            out = _run_op(op, target, unit, store, case_id, state)
            rec["oracle_output"] = out
            oracle_outputs[op] = out
        except Exception as e:
            rec["error"] = str(e)
            errors.append(f"step {i} ({op}): {e}")
        steps_out.append(rec)

    final = _final_answer(answer_from, steps_out)
    return {"steps": steps_out, "oracle_outputs": oracle_outputs, "final_answer": final, "errors": errors}


def make_prediction(case, task, target, plan_result, exec_result, mode, llm_model) -> dict:
    """Assemble the full prediction record saved to disk (the audit log)."""
    errors = list(exec_result.get("errors", []))
    if plan_result.get("error"):
        errors.append(plan_result["error"])
    return {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "task_name": task.name,
        "question": task.question_for(target),
        "target": target,
        "llm_model": llm_model,
        "mode": mode,
        "prompt": plan_result.get("prompt"),
        "raw_response": plan_result.get("raw_response"),
        "plan": plan_result.get("plan"),
        "plan_valid_json": plan_result.get("plan_valid_json", False),
        "steps": exec_result.get("steps", []),
        "oracle_outputs": exec_result.get("oracle_outputs", {}),
        "final_answer": exec_result.get("final_answer"),
        "errors": errors,
    }
