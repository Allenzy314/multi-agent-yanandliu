# Zero-Shot Oracle Ultrasound Agent

A baseline agent for ultrasound image understanding that **decouples reasoning from
perception**. An LLM plans *what* to do; a set of *oracle* tools simulate perfect
perception by reading ground-truth annotations; deterministic post-processing turns
the oracle outputs into a final answer.

## Overview & Design Principle

The agent never lets the language model see the answer:

- **The LLM sees only** the `image_id`, the natural-language `question`, the list of
  available `operations` (with one-line help), and the plan JSON schema. It never
  sees the image pixels or the ground-truth annotation. Its sole job is to emit a
  workflow **plan** — an ordered list of operations to run.
- **The oracle simulates perception.** Instead of a real detector / segmentator /
  classifier, the oracle reads the ground-truth annotation and returns what a
  *perfect* model would output (boxes, masks, labels). Every oracle response is
  tagged with `source = "ground_truth_annotation"`. This isolates the quality of the
  *agent's plan and post-processing* from the (here, assumed perfect) perception.
- **The executor + deterministic post-processing** run the planned operations
  against the oracle and compute the final answer (e.g. area from a mask, max
  diameter from a box, count from a detection list). No learned components are
  involved past the plan.
- **The rule-based baseline bypasses the LLM entirely.** For each task it emits the
  canonical plan directly. It is a sanity check: with a perfect oracle and a correct
  plan, the pipeline should score perfectly, so any error there is a bug in the
  executor / post-processing rather than the model.

Because the oracle is perfect, this is an **upper bound / debugging harness**: it
measures whether the agent chooses the right operations and post-processes them
correctly, not whether a real perception model would succeed.

## Module Map

| Path | Responsibility |
|------|----------------|
| `schemas.py` | `OPERATIONS`, `MEASURE_OPS`, `OP_HELP`, `UNITS`, answer-type constants, `ORACLE_SOURCE`; pydantic `Plan` / `PlanStep` / `FinalAnswer`; `PLAN_JSON_SCHEMA`; `extract_json`, `parse_and_validate_plan`. |
| `oracle/annotation_store.py` | `AnnotationStore.from_file(...)`; `Case` / `OracleObject` dataclasses; case lookup, object/measurement/target accessors, path resolution. |
| `oracle/oracle_detector.py` | `detect(store, cid, target)` → boxes/labels/scores from ground truth. |
| `oracle/oracle_segmentator.py` | `segment(store, cid, target)` → mask paths/labels from ground truth. |
| `oracle/oracle_classifier.py` | `classify(store, cid, target)` → label + degenerate probs from ground truth. |
| `postprocess/measurement.py` | Deterministic geometry: `load_mask`, `mask_area`, `bbox_from_mask`, `bbox_width_height`, `max_diameter_from_bbox`, `bbox_perimeter`, `count_objects`, `to_units`, ... |
| `agent/tasks.py` | `Task` definition + `TASKS` registry; question templates; `get_task`, `applicable`, `ground_truth`, `target_for_task`. |
| `agent/llm_client.py` | `get_llm_client(model)` → `MockLLMClient` for `mock`/`offline`/`none`/`""`, else `AnthropicClient`; `.complete(system, prompt)`. |
| `agent/planner.py` | `SYSTEM_PROMPT`, `build_planner_prompt(...)`, `plan(case, task, llm_client)` → prompt + raw response + validated plan. |
| `agent/executor.py` | `execute_plan(...)` runs a plan against the oracle + post-processing; `make_prediction(...)` builds the per-case prediction record. |
| `agent/rule_baseline.py` | `plan(case, task)` → deterministic canonical plan (same shape as `planner.plan`), no LLM. |
| `eval/evaluate_oracle_agent.py` | Scores prediction records against ground truth per task. |
| `scripts/make_synthetic_fixture.py` | Generate a tiny synthetic unified-annotation dataset for tests/smoke runs. |
| `scripts/run_oracle_agent.py` | Run the agent (rule / mock / real LLM) over an annotation file and write prediction records. |
| `scripts/build_unified_annotations.py` | Convert real FLARE ultrasound datasets into the unified annotation format. |

All imports are absolute from the repo root. Entry-point scripts insert the repo
root onto `sys.path` before importing project modules, so run `python` from the
repo root.

## Unified Annotation Format

The annotation file is a JSON **list** of case records. Paths are relative to the
annotation file, so `AnnotationStore.from_file(path)` resolves them without an extra
`--image_dir`. Example record (as produced by the synthetic fixture):

```json
{
  "case_id": "case_000000",
  "dataset": "synthetic_thyroid_nodule",
  "modality": "ultrasound",
  "image_path": "images/case_000000.png",
  "image_size": [160, 128],
  "spacing": null,
  "objects": [
    {
      "name": "thyroid_nodule",
      "class_label": "benign",
      "bbox": [10, 22, 58, 74],
      "mask_path": "masks/case_000000_thyroid_nodule_0.png",
      "contour": null
    }
  ],
  "image_label": "benign",
  "measurements": {}
}
```

- `image_size` is `[width, height]`; `bbox` is `[x_min, y_min, x_max, y_max]`.
- `spacing` is `null` or `[sx, sy]` (mm/pixel). When falsy, measurements stay in
  pixels; `to_units(..., unit)` only converts to `mm`/`mm2` when spacing is present.
- A case may hold multiple `objects` (the synthetic fixture gives two objects when
  `i % 3 == 0`).

## Plan JSON Schema

The LLM must emit a single JSON object matching `PLAN_JSON_SCHEMA`:

```json
{
  "type": "object",
  "properties": {
    "image_id": {"type": "string"},
    "target":   {"type": "string"},
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "op":     {"type": "string", "enum": ["detect", "segment", "classify",
                                                "measure_area", "measure_max_diameter",
                                                "measure_count", "measure_circumference"]},
          "target": {"type": "string"},
          "unit":   {"type": "string", "enum": ["mm", "mm2", "pixel"]}
        },
        "required": ["op"]
      }
    },
    "answer_from": {"type": "string"}
  },
  "required": ["image_id", "target", "steps"]
}
```

`answer_from` names the op (or a step index as a string) that produces the final
answer. Example plan for the `measure_area` task:

```json
{
  "image_id": "case_000000",
  "target": "thyroid_nodule",
  "steps": [
    {"op": "segment",      "target": "thyroid_nodule"},
    {"op": "measure_area", "target": "thyroid_nodule", "unit": "pixel"}
  ],
  "answer_from": "measure_area"
}
```

## Tasks

Defined in `agent/tasks.py` (`TASKS`); pick one with `--task_name`.

| Task | Operations | Answer type |
|------|------------|-------------|
| `detect` | `detect` | `bbox` `[x_min, y_min, x_max, y_max]` |
| `classify` | `classify` | `label` |
| `count` | `detect`, `measure_count` | `count` (int) |
| `measure_area` | `segment`, `measure_area` | `number` (pixels) |
| `measure_max_diameter` | `detect`, `measure_max_diameter` | `number` (pixels, diagonal of bbox) |

## Quickstart

Run everything from the repo root. `anthropic` is **not installed by default** — the
LLM client is lazy-imported, so use `--llm_model mock` (or `rule`) for fully offline
runs.

```bash
# 1. Generate a synthetic fixture (annotation.json + images/ + masks/)
python scripts/make_synthetic_fixture.py --out /tmp/fix --n 8

# 2. Run the deterministic rule baseline (no LLM)
python scripts/run_oracle_agent.py \
    --annotation_file /tmp/fix/annotation.json \
    --task_name detect --llm_model rule \
    --output_dir /tmp/preds_detect

# 3. Run the mock-LLM agent (offline; canned/deterministic responses)
python scripts/run_oracle_agent.py \
    --annotation_file /tmp/fix/annotation.json \
    --task_name detect --llm_model mock \
    --output_dir /tmp/preds_detect

# 4. Run a real LLM (requires `pip install anthropic` and ANTHROPIC_API_KEY)
python scripts/run_oracle_agent.py \
    --annotation_file /tmp/fix/annotation.json \
    --task_name detect --llm_model claude-opus-4-8 \
    --output_dir /tmp/preds_detect

# 5. Evaluate predictions against ground truth
python eval/evaluate_oracle_agent.py \
    --predictions_dir /tmp/preds_detect \
    --annotation_file /tmp/fix/annotation.json \
    --task_name detect

# 6. Build unified annotations from real FLARE data
python scripts/build_unified_annotations.py \
    --dataset_dir flare_dataset/Ultrasound/Dataset506_Thyroid_Nodule \
    --out /tmp/thyroid/annotation.json

# 7. Run the tests
python -m pytest -q
```

## Evaluation Metrics

`eval/evaluate_oracle_agent.py` scores the prediction records against ground truth
(`agent.tasks.ground_truth`), reporting metrics appropriate to each task's answer
type, plus pipeline-health metrics:

- **`detect` (bbox):** mean IoU between predicted and ground-truth boxes.
- **`classify` (label):** accuracy (fraction of exactly matching labels).
- **`count` (count):** exact-match accuracy and mean absolute error of the count.
- **`measure_area` / `measure_max_diameter` (number):** mean absolute error and
  mean relative error vs. the ground-truth value.
- **Pipeline health:** number of cases scored, plan-valid-JSON rate, and the count
  of cases that produced a usable `final_answer` (vs. `errors`).

With the perfect oracle, the `rule` baseline is expected to score perfectly on every
task; deviations indicate a bug in the executor or post-processing.

## Prediction Record

`scripts/run_oracle_agent.py` writes one JSON record per case (via
`executor.make_prediction`):

```json
{
  "case_id": "...", "dataset": "...", "task_name": "...", "question": "...",
  "target": "...", "llm_model": "...", "mode": "llm|rule",
  "prompt": "...", "raw_response": "...",
  "plan": { ... } , "plan_valid_json": true,
  "steps": [ {"index": 0, "op": "...", "target": "...", "unit": "...",
              "oracle_output": { ... }, "error": null} ],
  "oracle_outputs": { "op": { ... } },
  "final_answer": {"answer_type": "...", "value": ..., "unit": "..."},
  "errors": []
}
```

`mode` is `"rule"` for the rule baseline and `"llm"` otherwise; for the rule
baseline `prompt` and `raw_response` are `null`.
