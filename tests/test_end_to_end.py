"""End-to-end tests for the zero-shot Oracle ultrasound agent stack.

Everything runs offline against a synthetic fixture built by
``scripts.make_synthetic_fixture.build`` — no network, no real dataset, and the
mock LLM backend (``--llm_model mock``) so the ``anthropic`` package is never
imported.

The suite exercises the full pipeline: fixture -> annotation store -> oracle
backends -> measurement post-processing -> planner (mock) / rule baseline ->
executor -> prediction records -> evaluation metrics. It also pins down the
critical *LLM-isolation invariant*: the planner prompt never leaks ground-truth
outputs.
"""
from __future__ import annotations

import json
import math
import os
import re

import pytest

from schemas import (
    ANSWER_BBOX,
    ANSWER_COUNT,
    ANSWER_LABEL,
    ANSWER_NUMBER,
    ORACLE_SOURCE,
    extract_json,
    parse_and_validate_plan,
)
from oracle.annotation_store import AnnotationStore
from oracle import oracle_classifier, oracle_detector, oracle_segmentator
from postprocess import measurement as M
from agent import executor, planner, rule_baseline, tasks
from agent.llm_client import MockLLMClient, get_llm_client
from eval.evaluate_oracle_agent import evaluate
from scripts.make_synthetic_fixture import build

N_CASES = 8
TASK_NAMES = ["detect", "classify", "count", "measure_area", "measure_max_diameter"]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> str:
    """Build the synthetic dataset once for the whole module."""
    out = tmp_path_factory.mktemp("synthetic_fixture")
    ann_path = build(str(out), N_CASES)
    assert os.path.isfile(ann_path), "build() must write annotation.json"
    return str(out)


@pytest.fixture(scope="module")
def ann_path(fixture_dir) -> str:
    return os.path.join(fixture_dir, "annotation.json")


@pytest.fixture(scope="module")
def store(ann_path) -> AnnotationStore:
    return AnnotationStore.from_file(ann_path)


@pytest.fixture(scope="module")
def mock_client() -> MockLLMClient:
    return get_llm_client("mock")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _first_applicable_case(store: AnnotationStore, task):
    for case in store.cases():
        if tasks.applicable(task, case):
            return case
    raise AssertionError(f"no applicable case for task {task.name}")


def _answers_equal(answer_type, predicted, expected) -> bool:
    """Compare a predicted final-answer value against the ground-truth value."""
    if answer_type == ANSWER_BBOX:
        return list(predicted) == list(expected)
    if answer_type == ANSWER_LABEL:
        return predicted == expected
    if answer_type == ANSWER_COUNT:
        return int(predicted) == int(expected)
    if answer_type == ANSWER_NUMBER:
        return math.isclose(float(predicted), float(expected), rel_tol=1e-6, abs_tol=1e-9)
    raise AssertionError(f"unexpected answer_type {answer_type!r}")


def _run_case_task(store, case, task, plan_result):
    """Execute a plan_result end-to-end and return the executor result."""
    return executor.execute_plan(plan_result["plan"], store, case.case_id)


# --------------------------------------------------------------------------- #
# 1. Fixture + store
# --------------------------------------------------------------------------- #
def test_fixture_builds_and_store_loads(store):
    assert len(store) == N_CASES, f"expected {N_CASES} cases, got {len(store)}"
    assert len(store.case_ids()) == N_CASES

    case = store.get_case(store.case_ids()[0])
    assert case.image_size is not None and len(case.image_size) == 2, "image_size must be [w, h]"
    assert case.objects, "case must have at least one object"

    obj = case.objects[0]
    assert obj.bbox is not None and len(obj.bbox) == 4, "object must carry a 4-tuple bbox"
    assert obj.mask_path is not None, "object must carry a mask_path"
    assert os.path.isfile(store.resolve_path(obj.mask_path)), "mask file must exist on disk"


# --------------------------------------------------------------------------- #
# 2. Oracle backends
# --------------------------------------------------------------------------- #
def test_oracle_detect_matches_annotation(store):
    cid = store.case_ids()[0]
    case = store.get_case(cid)
    target = case.objects[0].name

    out = oracle_detector.detect(store, cid, target)
    assert set(out) >= {"boxes", "labels", "scores", "source"}
    assert out["source"] == ORACLE_SOURCE == "ground_truth_annotation"

    expected_boxes = [list(o.bbox) for o in store.get_objects(cid, target) if o.bbox is not None]
    assert out["boxes"] == expected_boxes, "detect must return the annotation bboxes verbatim"
    assert out["scores"] == [1.0] * len(expected_boxes), "oracle scores are all 1.0"


def test_oracle_segment_returns_mask_paths(store):
    cid = store.case_ids()[0]
    case = store.get_case(cid)
    target = case.objects[0].name

    out = oracle_segmentator.segment(store, cid, target)
    assert set(out) >= {"mask_paths", "labels", "source"}
    assert out["source"] == ORACLE_SOURCE
    expected = [o.mask_path for o in store.get_objects(cid, target) if o.mask_path is not None]
    assert out["mask_paths"] == expected
    for p in out["mask_paths"]:
        assert os.path.isfile(store.resolve_path(p)), f"mask path must resolve: {p}"


def test_oracle_classify_matches_label(store):
    cid = store.case_ids()[0]
    case = store.get_case(cid)
    target = case.objects[0].name

    out = oracle_classifier.classify(store, cid, target)
    assert set(out) >= {"label", "probs", "source"}
    assert out["source"] == ORACLE_SOURCE
    expected = case.objects[0].class_label
    assert out["label"] == expected, "classify must return the ground-truth class label"
    # in the fixture the image label mirrors the object class label
    assert out["label"] == case.image_label
    assert out["probs"] == {expected: 1.0}


# --------------------------------------------------------------------------- #
# 3. Measurement post-processing
# --------------------------------------------------------------------------- #
def test_measurement_functions_on_fixture(store):
    import numpy as np

    cid = store.case_ids()[0]
    case = store.get_case(cid)
    obj = case.objects[0]

    mask = M.load_mask(store.resolve_path(obj.mask_path))
    assert M.mask_area(mask) == int(np.count_nonzero(mask)), "mask_area == count of nonzero pixels"

    w, h = M.bbox_width_height(obj.bbox)
    assert math.isclose(
        M.max_diameter_from_bbox(obj.bbox), math.hypot(w, h), rel_tol=1e-9
    ), "max_diameter_from_bbox == hypot(w, h)"

    # spacing None => pixels unchanged, regardless of requested unit
    assert M.to_units(123.0, None, "length", "mm") == 123.0
    assert M.to_units(123.0, None, "area", "mm2") == 123.0
    assert M.to_units(123.0, case.spacing, "length", "pixel") == 123.0

    mb = M.mask_bbox(mask)
    assert mb is not None
    bx0, by0, bx1, by1 = obj.bbox
    # tight mask bbox must sit within the recorded bbox (far edges are exclusive,
    # hence the +1 tolerance on x_max/y_max)
    assert mb[0] >= bx0 and mb[1] >= by0
    assert mb[2] <= bx1 + 1 and mb[3] <= by1 + 1


# --------------------------------------------------------------------------- #
# 4. Planner (mock) produces valid plans with the right operations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_planner_mock_plan_valid_and_ops_match(store, mock_client, task_name):
    task = tasks.get_task(task_name)
    case = _first_applicable_case(store, task)

    plan_result = planner.plan(case, task, mock_client)
    assert plan_result["plan_valid_json"] is True, f"{task_name}: plan must be valid JSON"
    assert plan_result["error"] is None, f"{task_name}: planner error {plan_result['error']!r}"

    plan = plan_result["plan"]
    ops = [step["op"] for step in plan["steps"]]
    assert ops == list(task.expected_ops), f"{task_name}: ops {ops} != {list(task.expected_ops)}"


# --------------------------------------------------------------------------- #
# 5. Executor end-to-end: rule baseline AND mock planner match ground truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_end_to_end_answers_match_ground_truth(store, mock_client, task_name):
    task = tasks.get_task(task_name)
    case = _first_applicable_case(store, task)

    gt = tasks.ground_truth(task, case, store)
    assert gt is not None, f"{task_name}: ground truth must exist for an applicable case"

    # (a) rule baseline
    rule_plan = rule_baseline.plan(case, task)
    rule_exec = _run_case_task(store, case, task, rule_plan)
    assert rule_exec["errors"] == [], f"{task_name} rule: executor errors {rule_exec['errors']}"
    assert rule_plan["error"] is None
    rfa = rule_exec["final_answer"]
    assert rfa is not None, f"{task_name} rule: no final answer"
    assert rfa["answer_type"] == gt["answer_type"]
    assert _answers_equal(gt["answer_type"], rfa["value"], gt["value"]), (
        f"{task_name} rule: {rfa['value']!r} != {gt['value']!r}"
    )

    # (b) mock LLM planner
    llm_plan = planner.plan(case, task, mock_client)
    llm_exec = _run_case_task(store, case, task, llm_plan)
    assert llm_exec["errors"] == [], f"{task_name} llm: executor errors {llm_exec['errors']}"
    assert llm_plan["error"] is None
    lfa = llm_exec["final_answer"]
    assert lfa is not None, f"{task_name} llm: no final answer"
    assert lfa["answer_type"] == gt["answer_type"]
    assert _answers_equal(gt["answer_type"], lfa["value"], gt["value"]), (
        f"{task_name} llm: {lfa['value']!r} != {gt['value']!r}"
    )


# --------------------------------------------------------------------------- #
# 6. LLM-isolation invariant: the prompt must NOT leak ground truth
# --------------------------------------------------------------------------- #
def test_prompt_does_not_leak_classify_label(store, mock_client):
    task = tasks.get_task("classify")
    case = _first_applicable_case(store, task)
    plan_result = planner.plan(case, task, mock_client)
    prompt = plan_result["prompt"]

    label = case.objects[0].class_label
    assert label is not None
    assert label not in prompt, "planner prompt must not contain the class label"

    # positive checks: the prompt carries only what the LLM is allowed to see
    assert case.case_id in prompt, "prompt must include image_id"
    target = tasks.target_for_task(task, case)
    assert target in prompt, "prompt must include the target name"
    assert "classify" in prompt, "prompt must list operation names"


def test_prompt_does_not_leak_detect_bbox(store, mock_client):
    task = tasks.get_task("detect")
    case = _first_applicable_case(store, task)
    plan_result = planner.plan(case, task, mock_client)
    prompt = plan_result["prompt"]

    bbox = case.objects[0].bbox
    # maximal digit runs in the prompt: a coord "leaks" only if it appears as a
    # standalone integer token (surrounded by non-digits).
    digit_runs = set(re.findall(r"\d+", prompt))
    coords = [str(int(c)) for c in bbox]
    assert not all(c in digit_runs for c in coords), (
        f"planner prompt leaks the full bbox {bbox}"
    )

    # positive checks
    assert case.case_id in prompt
    target = tasks.target_for_task(task, case)
    assert target in prompt
    for op in task.expected_ops:
        assert op in prompt, f"prompt must list operation {op!r}"


# --------------------------------------------------------------------------- #
# 7. Evaluation over a directory of rule-mode prediction records
# --------------------------------------------------------------------------- #
def _dump_rule_predictions(store, pred_dir) -> int:
    """Generate + dump one rule-mode prediction record per (case, task)."""
    os.makedirs(pred_dir, exist_ok=True)
    n = 0
    for case in store.cases():
        for task_name in TASK_NAMES:
            task = tasks.get_task(task_name)
            if not tasks.applicable(task, case):
                continue
            target = tasks.target_for_task(task, case)
            plan_result = rule_baseline.plan(case, task)
            exec_result = executor.execute_plan(plan_result["plan"], store, case.case_id)
            rec = executor.make_prediction(
                case, task, target, plan_result, exec_result, mode="rule", llm_model="mock"
            )
            path = os.path.join(pred_dir, f"{case.case_id}__{task_name}.json")
            with open(path, "w") as f:
                json.dump(rec, f)
            n += 1
    return n


# Which evaluator metric carries the answer accuracy for each task's answer type.
_ANSWER_METRIC_BY_TASK = {
    "detect": "bbox_exact_match",
    "classify": "classification_exact_match",
    "count": "numeric_answer_accuracy",
    "measure_area": "numeric_answer_accuracy",
    "measure_max_diameter": "numeric_answer_accuracy",
}


def test_eval_rule_baseline_is_perfect(store, ann_path, tmp_path):
    pred_dir = os.path.join(str(tmp_path), "predictions")
    count = _dump_rule_predictions(store, pred_dir)
    assert count > 0, "must have generated prediction records"

    result = evaluate(pred_dir, ann_path)
    assert result["n_predictions"] == count
    assert result["n_skipped"] == 0

    metrics = result["metrics"]
    assert metrics["valid_json_rate"]["value"] == 1.0, metrics
    assert metrics["correct_operation_selection_rate"]["value"] == 1.0, metrics
    assert metrics["target_selection_accuracy"]["value"] == 1.0, metrics
    assert metrics["final_answer_schema_valid_rate"]["value"] == 1.0, metrics


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_eval_answer_accuracy_per_task(store, ann_path, tmp_path, task_name):
    pred_dir = os.path.join(str(tmp_path), "predictions_task")
    _dump_rule_predictions(store, pred_dir)

    result = evaluate(pred_dir, ann_path, task_name=task_name)
    assert result["n_predictions"] > 0, f"{task_name}: no predictions evaluated"

    metric_name = _ANSWER_METRIC_BY_TASK[task_name]
    metric = result["metrics"][metric_name]
    assert metric["value"] == 1.0, (
        f"{task_name}: {metric_name} = {metric['value']} != 1.0 for the rule baseline"
    )


# --------------------------------------------------------------------------- #
# 8. extract_json / parse_and_validate_plan
# --------------------------------------------------------------------------- #
def test_extract_json_from_fenced_block():
    text = "Here is the plan:\n```json\n{\"image_id\": \"c\", \"a\": 1}\n```\nDone."
    obj = extract_json(text)
    assert obj == {"image_id": "c", "a": 1}


def test_extract_json_from_prose():
    text = 'blah blah {"image_id": "c", "b": 2} trailing prose'
    obj = extract_json(text)
    assert obj == {"image_id": "c", "b": 2}


def test_extract_json_returns_none_when_absent():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None


def test_parse_and_validate_valid_plan():
    text = json.dumps(
        {
            "image_id": "case_000000",
            "target": "thyroid_nodule",
            "steps": [{"op": "detect", "target": "thyroid_nodule", "unit": "pixel"}],
            "answer_from": "detect",
        }
    )
    raw, plan_dict, err = parse_and_validate_plan(text)
    assert err is None
    assert raw is not None
    assert plan_dict is not None
    assert [s["op"] for s in plan_dict["steps"]] == ["detect"]


def test_parse_and_validate_plan_no_steps():
    # A plan whose steps list is empty must not validate (the locked schema
    # requires at least one step). ``raw_obj`` is still returned for auditing.
    text = json.dumps({"image_id": "case_000000", "target": "thyroid_nodule", "steps": []})
    raw, plan_dict, err = parse_and_validate_plan(text)
    assert raw is not None, "raw parsed object should still be returned"
    assert plan_dict is None, "a plan with no steps must not validate"
    assert err, "an error message must be reported"
