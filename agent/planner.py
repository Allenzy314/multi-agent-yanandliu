"""LLM planner: builds the prompt, calls the LLM, and parses a JSON workflow plan.

The LLM sees only: image_id, the task question, the target name, the available
operation names, and the plan JSON schema. It must return JSON only and must not
emit visual predictions (boxes / masks / labels / measurements) — only the plan.
"""
from __future__ import annotations

import json

from agent import tasks as _tasks
from schemas import OP_HELP, OPERATIONS, PLAN_JSON_SCHEMA, parse_and_validate_plan

SYSTEM_PROMPT = (
    "You are a medical-imaging workflow planner for ultrasound analysis. "
    "You DO NOT analyze the image yourself and you never invent bounding boxes, "
    "masks, labels, or measurements. You only decide which tool operations to "
    "run and in what order, then emit a JSON workflow plan. A separate executor "
    "runs the operations against trusted backends and produces the answer. "
    "Respond with a single JSON object and nothing else."
)


def build_planner_prompt(image_id, question, target, operations=OPERATIONS, op_help=OP_HELP) -> str:
    lines = [
        "You are given one ultrasound imaging task. Produce a JSON workflow plan.",
        "",
        f"image_id: {image_id}",
        f"target: {target}",
        f"question: {question}",
        "",
        "available_operations:",
        *[f"  - {op}: {op_help.get(op, '')}" for op in operations],
        "",
        "Output JSON schema:",
        json.dumps(PLAN_JSON_SCHEMA, indent=2),
        "",
        "Rules:",
        "- Use only operation names from available_operations.",
        "- Measurement operations (those starting with measure_) must be "
        "preceded by a detect or segment step that produces their geometry.",
        '- Set each step\'s "target" to the exact target string above.',
        "- Do NOT include any predictions (boxes, masks, labels, or numbers) in "
        "your output — only the plan of operations.",
        "- Respond with the JSON object only, no prose and no code fences.",
    ]
    return "\n".join(lines)


def plan(case, task, llm_client) -> dict:
    """Ask the LLM for a plan for ``case`` under ``task``.

    Returns a *plan result* dict::
        {"prompt", "raw_response", "plan", "plan_valid_json", "error"}
    ``plan`` is a validated ``Plan.model_dump()`` when possible, otherwise the
    raw parsed object (so the executor can still attempt best-effort execution),
    or None if nothing parseable came back.
    """
    target = _tasks.target_for_task(task, case)
    question = task.question_for(target)
    prompt = build_planner_prompt(case.case_id, question, target)
    result = {
        "prompt": prompt,
        "raw_response": None,
        "plan": None,
        "plan_valid_json": False,
        "error": None,
    }
    try:
        raw = llm_client.complete(SYSTEM_PROMPT, prompt)
    except Exception as e:
        result["error"] = f"llm error: {e}"
        return result

    result["raw_response"] = raw
    obj, plan_dict, err = parse_and_validate_plan(raw)
    result["plan_valid_json"] = obj is not None
    if plan_dict is not None:
        result["plan"] = plan_dict
    else:
        result["plan"] = obj  # keep raw obj for best-effort execution
        result["error"] = err
    return result
