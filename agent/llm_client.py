"""LLM client abstraction for the planner.

Two backends:
  * ``AnthropicClient`` — real Claude planner (lazy-imports ``anthropic``;
    defaults to ``claude-opus-4-8``; forces JSON via structured outputs with a
    prompt-only fallback).
  * ``MockLLMClient`` — offline stand-in that derives a plan JSON *from the
    prompt only* (never the annotation), so the full parse→execute path runs
    without a network or API key.

The mock deliberately parses the same labelled prompt the real planner sends, so
swapping backends changes nothing downstream.
"""
from __future__ import annotations

import json
import re
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-8"
MODEL_ALIASES = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
    "claude": "claude-opus-4-8",
}
_MOCK_NAMES = {"mock", "offline", "none", ""}


class BaseLLMClient:
    name = "base"

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Mock (offline) client
# --------------------------------------------------------------------------- #
def _field(prompt: str, key: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", prompt, re.MULTILINE)
    return m.group(1).strip() if m else None


def _guess_ops(question: str) -> list:
    q = (question or "").lower()
    if "how many" in q or "count" in q or "number of" in q:
        return ["detect", "measure_count"]
    if "area" in q:
        return ["segment", "measure_area"]
    if "circumference" in q or "perimeter" in q:
        return ["segment", "measure_circumference"]
    if "diameter" in q:
        return ["detect", "measure_max_diameter"]
    if "classif" in q or "class label" in q or "benign" in q or "malignant" in q:
        return ["classify"]
    if "segment" in q or "mask" in q:
        return ["segment"]
    # default: detection / bounding box
    return ["detect"]


class MockLLMClient(BaseLLMClient):
    name = "mock"

    def __init__(self, model: str = "mock"):
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        image_id = _field(prompt, "image_id") or "unknown"
        target = _field(prompt, "target")
        question = _field(prompt, "question") or ""
        ops = _guess_ops(question)
        steps = [{"op": op, "target": target, "unit": "pixel"} for op in ops]
        plan = {
            "image_id": image_id,
            "target": target,
            "steps": steps,
            "answer_from": ops[-1] if ops else None,
        }
        return json.dumps(plan)


# --------------------------------------------------------------------------- #
# Real Anthropic client
# --------------------------------------------------------------------------- #
def _text_of(resp) -> str:
    parts = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


class AnthropicClient(BaseLLMClient):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "the `anthropic` package is not installed; run `pip install "
                "anthropic` or use --llm_model mock"
            ) from e
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> str:
        from schemas import PLAN_JSON_SCHEMA

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Prefer structured outputs to force valid JSON; fall back to plain
        # prompting if the schema/param is rejected.
        try:
            resp = self._client.messages.create(
                output_config={"format": {"type": "json_schema", "schema": PLAN_JSON_SCHEMA}},
                **kwargs,
            )
        except Exception:
            resp = self._client.messages.create(**kwargs)
        return _text_of(resp)


def get_llm_client(model: Optional[str]) -> BaseLLMClient:
    m = (model or "").strip()
    if m.lower() in _MOCK_NAMES:
        return MockLLMClient(model=m or "mock")
    resolved = MODEL_ALIASES.get(m.lower(), m)
    return AnthropicClient(model=resolved)
