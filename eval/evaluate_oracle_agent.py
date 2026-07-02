"""Evaluate saved prediction JSONs against oracle ground truth.

Loads the prediction records produced by ``executor.make_prediction`` from a
directory and scores them against the deterministic ground truth computed by
``agent.tasks.ground_truth``. Exposes a reusable :func:`evaluate` function and a
CLI.

Metrics are each averaged only over the predictions for which they are defined,
and every metric reports its own denominator ``n``. Missing / ``None`` answers
count as failures rather than crashes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tasks  # noqa: E402
from oracle.annotation_store import AnnotationStore  # noqa: E402
from schemas import (  # noqa: E402
    ANSWER_BBOX,
    ANSWER_COUNT,
    ANSWER_LABEL,
    ANSWER_NUMBER,
)


def _norm(s) -> str:
    return (s or "").strip().lower() if isinstance(s, str) else ("" if s is None else str(s).strip().lower())


def iou(a, b) -> float:
    """Intersection-over-union of two ``[x0, y0, x1, y1]`` boxes.

    Returns 0.0 for malformed / degenerate inputs instead of raising.
    """
    try:
        ax0, ay0, ax1, ay1 = (float(v) for v in a)
        bx0, by0, bx1, by1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return 0.0
    ax0, ax1 = min(ax0, ax1), max(ax0, ax1)
    ay0, ay1 = min(ay0, ay1), max(ay0, ay1)
    bx0, bx1 = min(bx0, bx1), max(bx0, bx1)
    by0, by1 = min(by0, by1), max(by0, by1)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_equal(a, b) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6):
                return False
        except (TypeError, ValueError):
            return False
    return True


class _Metric:
    """Accumulator that averages a fraction over its own denominator."""

    def __init__(self):
        self.num = 0.0
        self.den = 0

    def add(self, hit: bool):
        self.den += 1
        if hit:
            self.num += 1.0

    def add_value(self, value: float):
        self.den += 1
        self.num += float(value)

    def result(self) -> dict:
        return {"value": (self.num / self.den if self.den else None), "n": self.den}


def _load_predictions(predictions_dir: str) -> list:
    preds = []
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.json"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        try:
            with open(path) as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict) or "case_id" not in obj:
            continue
        preds.append(obj)
    return preds


def evaluate(
    predictions_dir: str,
    annotation_file: str,
    image_dir: Optional[str] = None,
    task_name: Optional[str] = None,
) -> dict:
    """Score saved prediction JSONs against oracle ground truth.

    Returns a dict of the form::

        {"n_predictions": N, "n_skipped": K, "task": name_or_"mixed",
         "metrics": {metric: {"value": v, "n": denom}, ...},
         "numeric_mae": float|None}
    """
    store = AnnotationStore.from_file(annotation_file, image_dir)
    preds = _load_predictions(predictions_dir)

    valid_json = _Metric()
    op_selection = _Metric()
    target_selection = _Metric()
    schema_valid = _Metric()
    numeric_acc = _Metric()
    classification = _Metric()
    bbox_exact = _Metric()
    bbox_iou = _Metric()
    bbox_iou50 = _Metric()

    # Absolute errors kept per task so heterogeneous quantities (px^2 areas,
    # px diameters, unitless counts) are never blended into one MAE.
    numeric_abs_errors_by_task: dict = {}
    task_names_seen = set()
    n_used = 0
    n_skipped = 0

    for pred in preds:
        if task_name is not None and pred.get("task_name") != task_name:
            continue

        cid = pred.get("case_id")
        if cid not in store:
            n_skipped += 1
            continue
        case = store.get_case(cid)

        try:
            task = tasks.get_task(pred.get("task_name"))
        except KeyError:
            n_skipped += 1
            continue

        # A malformed annotation (e.g. a bad bbox) must not abort the whole run.
        try:
            gt = tasks.ground_truth(task, case, store)
        except Exception:
            n_skipped += 1
            continue
        if gt is None:
            n_skipped += 1
            continue

        n_used += 1
        task_names_seen.add(task.name)

        plan = pred.get("plan")
        final_answer = pred.get("final_answer")

        # valid_json_rate (over all used preds)
        valid_json.add(bool(pred.get("plan_valid_json")))

        # correct_operation_selection_rate
        if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
            ops = {s.get("op") for s in plan["steps"] if isinstance(s, dict)}
            op_selection.add(ops == set(task.expected_ops))
        else:
            op_selection.add(False)

        # target_selection_accuracy
        expected_target = tasks.target_for_task(task, case)
        target_selection.add(_norm(pred.get("target")) == _norm(expected_target))

        # final_answer_schema_valid_rate
        schema_ok = (
            isinstance(final_answer, dict)
            and final_answer.get("answer_type") == task.answer_type
            and final_answer.get("value") is not None
        )
        schema_valid.add(schema_ok)

        pred_value = final_answer.get("value") if isinstance(final_answer, dict) else None
        gt_value = gt.get("value")

        # answer-type-specific metrics
        if task.answer_type in (ANSWER_NUMBER, ANSWER_COUNT):
            hit = False
            if pred_value is not None:
                try:
                    if task.answer_type == ANSWER_COUNT:
                        hit = int(pred_value) == int(gt_value)
                    else:
                        hit = math.isclose(
                            float(pred_value), float(gt_value), rel_tol=1e-3, abs_tol=1e-3
                        )
                    numeric_abs_errors_by_task.setdefault(task.name, []).append(
                        abs(float(pred_value) - float(gt_value))
                    )
                except (TypeError, ValueError):
                    hit = False
            numeric_acc.add(hit)

        elif task.answer_type == ANSWER_LABEL:
            classification.add(pred_value is not None and _norm(pred_value) == _norm(gt_value))

        elif task.answer_type == ANSWER_BBOX:
            bbox_exact.add(_bbox_equal(pred_value, gt_value))
            score = iou(pred_value, gt_value) if pred_value is not None else 0.0
            bbox_iou.add_value(score)
            bbox_iou50.add(score >= 0.5)

    metrics = {
        "valid_json_rate": valid_json.result(),
        "correct_operation_selection_rate": op_selection.result(),
        "target_selection_accuracy": target_selection.result(),
        "final_answer_schema_valid_rate": schema_valid.result(),
        "numeric_answer_accuracy": numeric_acc.result(),
        "classification_exact_match": classification.result(),
        "bbox_exact_match": bbox_exact.result(),
        "bbox_mean_iou": bbox_iou.result(),
        "bbox_iou@0.5": bbox_iou50.result(),
    }

    numeric_mae_by_task = {
        t: (sum(errs) / len(errs) if errs else None)
        for t, errs in numeric_abs_errors_by_task.items()
    }
    # A single top-level MAE is only meaningful for one numeric quantity.
    numeric_mae = (
        next(iter(numeric_mae_by_task.values())) if len(numeric_mae_by_task) == 1 else None
    )

    if task_name is not None:
        task_label = task_name
    elif len(task_names_seen) == 1:
        task_label = next(iter(task_names_seen))
    else:
        task_label = "mixed"

    return {
        "n_predictions": n_used,
        "n_skipped": n_skipped,
        "task": task_label,
        "metrics": metrics,
        "numeric_mae": numeric_mae,
        "numeric_mae_by_task": numeric_mae_by_task,
    }


def _print_table(result: dict) -> None:
    print(f"task:          {result['task']}")
    print(f"n_predictions: {result['n_predictions']}")
    print(f"n_skipped:     {result['n_skipped']}")
    print("-" * 52)
    print(f"{'metric':<36}{'value':>10}{'n':>6}")
    print("-" * 52)
    for name, m in result["metrics"].items():
        v = m["value"]
        vs = "n/a" if v is None else f"{v:.4f}"
        print(f"{name:<36}{vs:>10}{m['n']:>6}")
    print("-" * 52)
    by_task = result.get("numeric_mae_by_task") or {}
    if by_task:
        for t, mae in by_task.items():
            label = f"numeric_mae[{t}]"
            print(f"{label:<36}{('n/a' if mae is None else f'{mae:.4f}'):>10}")
    else:
        mae = result.get("numeric_mae")
        print(f"{'numeric_mae':<36}{('n/a' if mae is None else f'{mae:.4f}'):>10}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", required=True, help="directory of prediction *.json files")
    parser.add_argument("--annotation_file", required=True, help="unified annotation JSON file")
    parser.add_argument("--image_dir", default=None, help="optional image/mask root for path resolution")
    parser.add_argument("--task_name", default=None, help="restrict evaluation to a single task")
    parser.add_argument("--out", default=None, help="optional path to write the metrics JSON")
    args = parser.parse_args(argv)

    result = evaluate(
        predictions_dir=args.predictions_dir,
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        task_name=args.task_name,
    )

    _print_table(result)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote metrics to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
