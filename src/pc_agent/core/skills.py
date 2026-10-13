"""Named reusable workflows saved as JSON skills."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pc_agent.config import skills_dir


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "skill"


def list_skills() -> list[dict]:
    out = []
    for p in sorted(skills_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["id"] = p.stem
            out.append(data)
        except Exception:
            continue
    return out


def save_skill(name: str, description: str, steps: list[dict]) -> dict:
    sid = _slug(name)
    data = {"name": name, "description": description, "steps": steps}
    path = skills_dir() / f"{sid}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["id"] = sid
    return data


def get_skill(skill_id: str) -> dict | None:
    path = skills_dir() / f"{_slug(skill_id)}.json"
    if not path.exists():
        # try exact stem
        path = skills_dir() / f"{skill_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["id"] = path.stem
    return data


def delete_skill(skill_id: str) -> bool:
    path = skills_dir() / f"{skill_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


DECLARATIONS = [
    {
        "name": "save_skill",
        "description": "Save a repeated workflow as a named reusable skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {action, args, expect} steps.",
                },
            },
            "required": ["name", "steps"],
        },
    },
    {
        "name": "list_skills",
        "description": "List saved skills.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "run_skill",
        "description": "Run a saved skill by id or name.",
        "parameters": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}},
            "required": ["skill_id"],
        },
    },
]


def make_handlers(planner=None) -> dict:
    def save_skill_h(args: dict) -> dict:
        data = save_skill(
            args.get("name") or "skill",
            args.get("description") or "",
            list(args.get("steps") or []),
        )
        return {"ok": True, "skill": data}

    def list_skills_h(_args: dict) -> dict:
        return {"ok": True, "skills": list_skills()}

    def run_skill_h(args: dict) -> dict:
        skill = get_skill(args.get("skill_id") or "")
        if not skill:
            return {"ok": False, "error": "skill not found"}
        if planner is None:
            return {"ok": False, "error": "planner unavailable", "skill": skill}
        from pc_agent.core.planner import Plan, PlanStep

        plan = Plan(
            goal=skill.get("name") or skill["id"],
            steps=[
                PlanStep(
                    id=i + 1,
                    action=s.get("action", ""),
                    args=s.get("args") or {},
                    expect=s.get("expect") or "",
                )
                for i, s in enumerate(skill.get("steps") or [])
            ],
        )
        result = planner.execute(plan)
        return {
            "ok": result.result == "ok",
            "result": result.result,
            "steps": [
                {"id": s.id, "action": s.action, "status": s.status, "detail": s.detail}
                for s in result.steps
            ],
        }

    return {
        "save_skill": save_skill_h,
        "list_skills": list_skills_h,
        "run_skill": run_skill_h,
    }
