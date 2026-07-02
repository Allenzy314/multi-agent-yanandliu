"""Shared data contracts for the zero-shot Oracle ultrasound agent.

Everything that must agree across the agent / oracle / eval layers lives here:
the operation vocabulary, the LLM *plan* schema, the *final answer* schema, and
robust JSON extraction. Keeping these in one place is what lets the planner,
executor, and evaluator interlock without drifting.

The LLM only ever sees: image_id, the task question, the operation names, and
the plan JSON schema. It never sees the ground-truth annotation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# --- Operation vocabulary the LLM may choose from ---------------------------
OP_DETECT = "detect"
OP_SEGMENT = "segment"
OP_CLASSIFY = "classify"
OP_MEASURE_AREA = "measure_area"
OP_MEASURE_MAX_DIAMETER = "measure_max_diameter"
OP_MEASURE_COUNT = "measure_count"
OP_MEASURE_CIRCUMFERENCE = "measure_circumference"

OPERATIONS = [
    OP_DETECT,
    OP_SEGMENT,
    OP_CLASSIFY,
    OP_MEASURE_AREA,
    OP_MEASURE_MAX_DIAMETER,
    OP_MEASURE_COUNT,
    OP_MEASURE_CIRCUMFERENCE,
]

# Measurement ops derive geometry from a prior detect/segment step.
MEASURE_OPS = {
    OP_MEASURE_AREA,
    OP_MEASURE_MAX_DIAMETER,
    OP_MEASURE_COUNT,
    OP_MEASURE_CIRCUMFERENCE,
}

# One-line descriptions handed to the LLM so it can pick operations.
OP_HELP = {
    OP_DETECT: "return bounding box(es) for the target structure",
    OP_SEGMENT: "return segmentation mask(s) for the target structure",
    OP_CLASSIFY: "return the class label for the target structure",
    OP_MEASURE_AREA: "compute the area of the target (needs a prior segment step)",
    OP_MEASURE_MAX_DIAMETER: "compute the maximum diameter of the target (needs a prior detect step)",
    OP_MEASURE_COUNT: "count how many target instances are present (needs a prior detect step)",
    OP_MEASURE_CIRCUMFERENCE: "compute the circumference of the target (needs a prior segment or detect step)",
}

# Units
UNIT_PIXEL = "pixel"
UNIT_MM = "mm"
UNIT_MM2 = "mm2"
UNITS = {UNIT_PIXEL, UNIT_MM, UNIT_MM2}

# Final-answer types
ANSWER_BBOX = "bbox"
ANSWER_LABEL = "label"
ANSWER_NUMBER = "number"
ANSWER_COUNT = "count"
ANSWER_TYPES = {ANSWER_BBOX, ANSWER_LABEL, ANSWER_NUMBER, ANSWER_COUNT}

# Provenance tag for every oracle response (these are ground-truth simulations).
ORACLE_SOURCE = "ground_truth_annotation"


class PlanStep(BaseModel):
    """One operation in the workflow plan."""

    op: str
    target: Optional[str] = None
    unit: str = UNIT_PIXEL

    @field_validator("op")
    @classmethod
    def _op_known(cls, v: str) -> str:
        if v not in OPERATIONS:
            raise ValueError(f"unknown op {v!r}; must be one of {OPERATIONS}")
        return v

    @field_validator("unit")
    @classmethod
    def _unit_known(cls, v: str) -> str:
        if v not in UNITS:
            raise ValueError(f"unknown unit {v!r}; must be one of {sorted(UNITS)}")
        return v


class Plan(BaseModel):
    """The JSON workflow plan the LLM must emit."""

    image_id: str
    target: Optional[str] = None
    steps: list[PlanStep] = Field(default_factory=list)
    # Which step produces the final answer: an op name or a step index (as str).
    answer_from: Optional[str] = None

    @field_validator("steps")
    @classmethod
    def _nonempty(cls, v: list) -> list:
        if not v:
            raise ValueError("plan must contain at least one step")
        return v


class FinalAnswer(BaseModel):
    """Deterministic post-processed answer derived from oracle outputs."""

    answer_type: str
    value: Any = None
    unit: Optional[str] = None

    @field_validator("answer_type")
    @classmethod
    def _atype(cls, v: str) -> str:
        if v not in ANSWER_TYPES:
            raise ValueError(f"unknown answer_type {v!r}")
        return v


# JSON schema for the plan — embedded in the prompt and (optionally) used to
# constrain the model via structured outputs.
PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "target": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": list(OPERATIONS)},
                    "target": {"type": "string"},
                    "unit": {"type": "string", "enum": sorted(UNITS)},
                },
                "required": ["op"],
                "additionalProperties": False,
            },
        },
        "answer_from": {"type": "string"},
    },
    "required": ["image_id", "target", "steps"],
    "additionalProperties": False,
}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _json_candidates(text: str):
    """Yield progressively looser candidate substrings that might be JSON."""
    yield text
    m = _FENCE_RE.search(text)
    if m:
        yield m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        yield text[start : end + 1]


def extract_json(text: Optional[str]) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from an LLM response.

    Handles ```json fenced blocks and surrounding prose. Returns None if no
    parseable object is found.
    """
    if not text:
        return None
    text = text.strip()
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_and_validate_plan(text: Optional[str]):
    """Parse + validate an LLM response into a plan.

    Returns ``(raw_obj, plan_dict, error)`` where ``raw_obj`` is the parsed JSON
    (or None if unparseable), ``plan_dict`` is a validated ``Plan.model_dump()``
    (or None), and ``error`` is a message on failure.
    """
    obj = extract_json(text)
    if obj is None:
        return None, None, "no valid JSON object found in response"
    try:
        plan = Plan.model_validate(obj)
    except Exception as e:  # pydantic ValidationError or otherwise
        return obj, None, f"plan schema invalid: {e}"
    return obj, plan.model_dump(), None
