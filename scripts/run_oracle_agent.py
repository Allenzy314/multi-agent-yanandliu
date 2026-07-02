"""Run the zero-shot oracle ultrasound agent over a dataset.

For each applicable case this builds a plan (via the LLM planner or the rule
baseline), executes it against the oracle backends, and writes one prediction
JSON per case to ``--output_dir`` plus a ``_summary.json`` roll-up.

Usage::

    python scripts/run_oracle_agent.py \\
        --annotation_file /tmp/fixture/annotation.json \\
        --task_name detect --llm_model mock --output_dir /tmp/preds
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import executor, planner, rule_baseline, tasks
from agent.llm_client import get_llm_client
from oracle.annotation_store import AnnotationStore

RULE_MODELS = {"rule", "rule_baseline"}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the oracle ultrasound agent over a dataset.")
    ap.add_argument("--annotation_file", required=True, help="unified annotation JSON file")
    ap.add_argument("--image_dir", default=None, help="optional image/mask root directory")
    ap.add_argument("--task_name", required=True, choices=sorted(tasks.TASKS.keys()), help="task to run")
    ap.add_argument("--llm_model", default="mock", help="LLM model id, or 'rule'/'rule_baseline'")
    ap.add_argument("--output_dir", required=True, help="directory for per-case prediction JSONs")
    ap.add_argument("--max_cases", type=int, default=None, help="limit number of applicable cases")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    store = AnnotationStore.from_file(args.annotation_file, args.image_dir)
    task = tasks.get_task(args.task_name)

    is_rule = args.llm_model in RULE_MODELS
    mode = "rule" if is_rule else "llm"
    llm = None if is_rule else get_llm_client(args.llm_model)

    os.makedirs(args.output_dir, exist_ok=True)

    n_cases = 0
    n_written = 0
    n_failed = 0

    for case_id in store.case_ids():
        case = store.get_case(case_id)
        if not tasks.applicable(task, case):
            continue
        if args.max_cases is not None and n_cases >= args.max_cases:
            break
        n_cases += 1

        try:
            target = tasks.target_for_task(task, case)
            if is_rule:
                plan_result = rule_baseline.plan(case, task)
            else:
                plan_result = planner.plan(case, task, llm)
            exec_result = executor.execute_plan(plan_result["plan"], store, case_id)
            pred = executor.make_prediction(
                case, task, target, plan_result, exec_result, mode, args.llm_model
            )
            # task in the filename so several tasks can share one --output_dir
            # without clobbering each other (the evaluator supports mixed dirs).
            out_name = f"{case_id}__{task.name}.json"
            with open(os.path.join(args.output_dir, out_name), "w") as f:
                json.dump(pred, f, indent=2)
            n_written += 1
            final = pred.get("final_answer")
            value = final.get("value") if isinstance(final, dict) else None
            print(f"[{n_cases}] {case_id}: {mode} -> {value}")
        except Exception as e:  # keep going on per-case failures
            n_failed += 1
            print(f"[{n_cases}] {case_id}: FAILED ({e})")

    summary = {
        "task_name": task.name,
        "llm_model": args.llm_model,
        "mode": mode,
        "n_cases": n_cases,
        "n_written": n_written,
        "n_failed": n_failed,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, f"_summary_{task.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"done: task={task.name} model={args.llm_model} mode={mode} "
        f"cases={n_cases} written={n_written} failed={n_failed} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
