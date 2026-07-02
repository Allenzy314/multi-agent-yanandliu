"""Rule-based oracle baseline that bypasses the LLM entirely.

Instead of asking an LLM to plan, it builds the canonical plan directly from the
task definition (the expected operation sequence + the resolved target). This is
the sanity-check upper bound: run through the *same* executor and it should score
essentially perfectly against the oracle ground truth. Any gap reveals a bug in
the executor / oracle / measurement stack rather than an LLM mistake.
"""
from __future__ import annotations

from agent import tasks as _tasks


def plan(case, task) -> dict:
    """Build a deterministic plan result (same shape the LLM planner returns)."""
    target = _tasks.target_for_task(task, case)
    steps = [{"op": op, "target": target, "unit": task.unit} for op in task.expected_ops]
    plan_dict = {
        "image_id": case.case_id,
        "target": target,
        "steps": steps,
        "answer_from": task.expected_ops[-1] if task.expected_ops else None,
    }
    return {
        "prompt": None,
        "raw_response": None,
        "plan": plan_dict,
        "plan_valid_json": True,
        "error": None,
    }
