"""Schedule one-shot or recurring tasks the agent runs itself."""

from __future__ import annotations

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "schedule_task",
        "description": (
            "Queue a task for later. Provide run_at as ISO-8601 UTC, or a cron "
            "expression for recurring work. The agent will execute the prompt when due."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "prompt": {
                    "type": "string",
                    "description": "What the agent should do when the task fires.",
                },
                "run_at": {"type": "string", "description": "ISO-8601 time (UTC)."},
                "cron": {
                    "type": "string",
                    "description": "5-field cron, e.g. '0 9 * * *' for 09:00 daily.",
                },
            },
            "required": ["title", "prompt"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List queued / recent agent tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "done", "failed", "cancelled"],
                }
            },
        },
    },
    {
        "name": "cancel_task",
        "description": "Cancel a pending task by id.",
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
]


def make_handler(name: str, memory=None, scheduler=None, **_kwargs):
    def schedule_task(args: dict) -> dict:
        if memory is None:
            return {"ok": False, "error": "memory unavailable"}
        tid = memory.enqueue_task(
            title=args.get("title") or "task",
            payload={"prompt": args.get("prompt") or ""},
            run_at=args.get("run_at"),
            cron=args.get("cron"),
        )
        if scheduler is not None and args.get("cron"):
            try:
                scheduler.add_cron_task(tid, args["cron"], args.get("prompt") or "")
            except Exception as exc:
                return {"ok": True, "task_id": tid, "cron_warning": str(exc)}
        safety.log_action("schedule", "enqueue", {"task_id": tid})
        return {"ok": True, "task_id": tid}

    def list_tasks(args: dict) -> dict:
        if memory is None:
            return {"ok": False, "error": "memory unavailable"}
        return {"ok": True, "tasks": memory.list_tasks(status=args.get("status"))}

    def cancel_task(args: dict) -> dict:
        if memory is None:
            return {"ok": False, "error": "memory unavailable"}
        tid = int(args["task_id"])
        memory.update_task(tid, status="cancelled")
        if scheduler is not None:
            try:
                scheduler.cancel(tid)
            except Exception:
                pass
        return {"ok": True, "task_id": tid}

    return {
        "schedule_task": schedule_task,
        "list_tasks": list_tasks,
        "cancel_task": cancel_task,
    }[name]
