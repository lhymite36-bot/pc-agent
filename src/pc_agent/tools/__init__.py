"""Desktop tool implementations and Gemini declarations."""

from __future__ import annotations

from typing import Any, Callable

from pc_agent.core.memory import MemoryStore
from pc_agent.core import plugins as plugin_loader
from pc_agent.core import skills as skills_mod
from pc_agent.tools import (
    apps,
    browser,
    clipboard,
    files,
    input_ctrl,
    schedule_tools,
    screen,
    shell,
    system,
    windows,
)


def build_tooling(
    memory: MemoryStore,
    scheduler=None,
    planner=None,
) -> tuple[list[dict], dict[str, Callable[[dict], Any]]]:
    declarations: list[dict] = []
    handlers: dict[str, Callable[[dict], Any]] = {}

    modules = [
        shell,
        files,
        apps,
        screen,
        input_ctrl,
        windows,
        clipboard,
        system,
        schedule_tools,
        browser,
    ]
    for mod in modules:
        for decl in mod.DECLARATIONS:
            declarations.append(decl)
            name = decl["name"]
            handlers[name] = mod.make_handler(
                name, memory=memory, scheduler=scheduler
            )

    # Skills
    for decl in skills_mod.DECLARATIONS:
        declarations.append(decl)
    handlers.update(skills_mod.make_handlers(planner=planner))

    # Memory tools
    for decl in memory.tool_specs():
        declarations.append(decl)
        name = decl["name"]
        handlers[name] = (lambda n: (lambda args: memory.dispatch_tool(n, args)))(name)

    # Planner entrypoint
    declarations.append(
        {
            "name": "run_plan",
            "description": (
                "Plan and execute a multi-step goal on this PC. "
                "Plans, runs tools, verifies with a screenshot, retries once on failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
            },
        }
    )

    def run_plan(args: dict) -> dict:
        if planner is None:
            return {"ok": False, "error": "planner unavailable"}
        plan = planner.plan(args.get("goal") or "")
        if not plan.steps:
            return {"ok": False, "error": plan.result or "empty plan"}
        result = planner.execute(plan)
        return {
            "ok": result.result == "ok",
            "result": result.result,
            "steps": [
                {
                    "id": s.id,
                    "action": s.action,
                    "status": s.status,
                    "detail": s.detail,
                }
                for s in result.steps
            ],
        }

    handlers["run_plan"] = run_plan

    # Speak tool (uses voice engine attached later via closure override in app)
    declarations.append(
        {
            "name": "speak",
            "description": "Speak a short reply aloud with the configured human voice.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    )
    handlers["speak"] = lambda args: {"ok": False, "error": "voice not wired"}

    # Plugins last (cannot override built-ins)
    p_decls, p_hands = plugin_loader.load_plugins()
    for d in p_decls:
        if d["name"] in handlers:
            continue
        declarations.append(d)
        handlers[d["name"]] = p_hands[d["name"]]

    return declarations, handlers
