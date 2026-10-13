"""Multi-step planner with execute → screenshot verify → retry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("pc_agent.planner")


@dataclass
class PlanStep:
    id: int
    action: str
    args: dict
    expect: str = ""
    status: str = "pending"  # pending|ok|failed|skipped
    detail: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    result: str = ""


class Planner:
    """Ask the model for a plan, run tools, verify with a screenshot, retry once."""

    def __init__(
        self,
        brain,
        tool_handlers: dict[str, Callable[[dict], Any]],
        take_screenshot: Callable[[], dict] | None = None,
    ) -> None:
        self.brain = brain
        self.tool_handlers = tool_handlers
        self.take_screenshot = take_screenshot

    def plan(self, goal: str) -> Plan:
        prompt = (
            "Break this goal into 2-8 concrete PC tool steps. "
            "Reply ONLY with JSON: "
            '{"steps":[{"action":"<tool_name>","args":{...},"expect":"what success looks like"}]}. '
            f"Goal: {goal}"
        )
        # Use a direct generate without polluting chat history too much
        try:
            from google.genai import types

            response = self.brain.client.models.generate_content(
                model=self.brain.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            text = _extract_json(text)
            data = json.loads(text)
            steps = []
            for i, s in enumerate(data.get("steps") or []):
                steps.append(
                    PlanStep(
                        id=i + 1,
                        action=s.get("action", ""),
                        args=s.get("args") or {},
                        expect=s.get("expect") or "",
                    )
                )
            return Plan(goal=goal, steps=steps)
        except Exception as exc:
            logger.warning("plan failed: %s", exc)
            return Plan(goal=goal, steps=[], result=f"plan error: {exc}")

    def execute(self, plan: Plan, max_retries: int = 1) -> Plan:
        for step in plan.steps:
            ok = self._run_step(step)
            if not ok and max_retries > 0:
                step.detail += " | retrying"
                ok = self._run_step(step)
            if not ok:
                plan.result = f"Stopped at step {step.id}: {step.detail}"
                return plan
            # Self-check via screenshot when expectation is visual
            if step.expect and self.take_screenshot is not None:
                shot = self.take_screenshot()
                if shot.get("ok") and self._verify(step, shot):
                    step.status = "ok"
                elif shot.get("ok"):
                    # one verify-driven retry
                    step.detail += " | verify failed, retry"
                    if not self._run_step(step):
                        plan.result = f"Verify failed at step {step.id}"
                        return plan
        plan.result = "ok"
        return plan

    def _run_step(self, step: PlanStep) -> bool:
        handler = self.tool_handlers.get(step.action)
        if handler is None:
            step.status = "failed"
            step.detail = f"unknown tool {step.action}"
            return False
        try:
            result = handler(step.args)
            step.detail = json.dumps(result)[:500]
            ok = bool(result.get("ok", True)) if isinstance(result, dict) else True
            step.status = "ok" if ok else "failed"
            return ok
        except Exception as exc:
            step.status = "failed"
            step.detail = str(exc)
            return False

    def _verify(self, step: PlanStep, shot: dict) -> bool:
        """Ask the model if the screenshot matches the expectation."""
        try:
            from google.genai import types
            from pathlib import Path

            path = shot.get("path")
            if not path or not Path(path).exists():
                return True
            data = Path(path).read_bytes()
            prompt = (
                f"Goal step expectation: {step.expect}\n"
                "Does this screenshot show the expectation was met? "
                'Answer ONLY JSON: {"met": true/false, "reason": "..."}.'
            )
            response = self.brain.client.models.generate_content(
                model=self.brain.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=data, mime_type="image/png"),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = _extract_json((response.text or "").strip())
            data_j = json.loads(text)
            return bool(data_j.get("met"))
        except Exception as exc:
            logger.debug("verify skipped: %s", exc)
            return True  # don't block on verify errors


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
