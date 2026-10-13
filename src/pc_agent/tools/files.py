"""Read / write / list files."""

from __future__ import annotations

import os
from pathlib import Path

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "list_files",
        "description": "List files and folders in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "glob": {"type": "string", "description": "Optional glob pattern, e.g. *.txt"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file (up to ~200KB).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a text file. Creates parent folders.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file or empty directory. Requires user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
]


def list_files(args: dict) -> dict:
    path = Path(args["path"]).expanduser()
    if not path.exists():
        return {"ok": False, "error": "path does not exist"}
    if not path.is_dir():
        return {"ok": False, "error": "not a directory"}
    pattern = args.get("glob") or "*"
    entries = []
    for p in sorted(path.glob(pattern))[:500]:
        try:
            st = p.stat()
            entries.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "is_dir": p.is_dir(),
                    "size": st.st_size if p.is_file() else None,
                }
            )
        except OSError:
            continue
    return {"ok": True, "entries": entries}


def read_file(args: dict) -> dict:
    path = Path(args["path"]).expanduser()
    max_chars = int(args.get("max_chars") or 200_000)
    if not path.is_file():
        return {"ok": False, "error": "not a file"}
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        truncated = len(data) > max_chars
        return {
            "ok": True,
            "content": data[:max_chars],
            "truncated": truncated,
            "size": path.stat().st_size,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def write_file(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    path = Path(args["path"]).expanduser()
    content = args.get("content", "")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        safety.log_action("files", "write", {"path": str(path), "bytes": len(content)})
        return {"ok": True, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def delete_file(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    path = Path(args["path"]).expanduser()
    if not safety.confirm("delete_file", f"Delete {path}?"):
        return {"ok": False, "error": "User denied confirmation"}
    try:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
        safety.log_action("delete_file", "delete", {"path": str(path)})
        return {"ok": True, "path": str(path)}
    except Exception as exc:
        safety.log_action("delete_file", "delete", {"path": str(path), "error": str(exc)}, ok=False)
        return {"ok": False, "error": str(exc)}


_HANDLERS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
}


def make_handler(name: str, **_kwargs):
    return _HANDLERS[name]
