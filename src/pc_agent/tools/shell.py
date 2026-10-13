"""Run PowerShell / cmd commands."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from pc_agent.core import safety

DECLARATIONS = [
    {
        "name": "run_shell",
        "description": (
            "Run a shell command on the PC. On Windows uses PowerShell by default; "
            "set shell='cmd' for cmd.exe. Returns stdout, stderr, and exit code. "
            "Do not use for deletions or installs without user intent — those need confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "shell": {
                    "type": "string",
                    "enum": ["powershell", "cmd", "bash"],
                    "description": "Shell to use (default powershell on Windows, bash elsewhere).",
                },
                "timeout_sec": {"type": "integer", "description": "Timeout seconds (default 60)."},
                "cwd": {"type": "string", "description": "Working directory."},
            },
            "required": ["command"],
        },
    }
]


def _is_dangerous_command(command: str) -> str | None:
    lower = command.lower()
    patterns = [
        ("remove-item", "delete_file"),
        ("rm -", "delete_file"),
        ("del ", "delete_file"),
        ("rmdir", "delete_file"),
        ("rd ", "delete_file"),
        ("format ", "delete_file"),
        ("install-package", "install_software"),
        ("winget install", "install_software"),
        ("choco install", "install_software"),
        ("pip install", "install_software"),
        ("shutdown", "shutdown"),
        ("restart-computer", "shutdown"),
        ("stop-computer", "shutdown"),
        ("reg add", "registry_edit"),
        ("reg delete", "registry_edit"),
        ("set-itemproperty.*hk", "registry_edit"),
    ]
    import re

    for pat, cat in patterns:
        if re.search(pat, lower):
            return cat
    return None


def run_shell(args: dict) -> dict:
    if safety.is_halted():
        return {"ok": False, "error": "Agent is halted (kill-switch)."}
    command = args.get("command") or ""
    if not command.strip():
        return {"ok": False, "error": "empty command"}

    cat = _is_dangerous_command(command)
    if cat and not safety.confirm(cat, f"Run: {command}"):
        return {"ok": False, "error": "User denied confirmation", "category": cat}

    shell_name = (args.get("shell") or ("powershell" if sys.platform == "win32" else "bash")).lower()
    timeout = int(args.get("timeout_sec") or 60)
    cwd = args.get("cwd") or None

    if shell_name == "powershell":
        argv = [
            "powershell" if sys.platform == "win32" else "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
    elif shell_name == "cmd":
        argv = ["cmd", "/c", command]
    else:
        argv = ["bash", "-lc", command]

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        safety.log_action(
            "shell",
            "run",
            {"command": command, "shell": shell_name, "code": proc.returncode},
            ok=proc.returncode == 0,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": (proc.stdout or "")[-8000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired:
        safety.log_action("shell", "timeout", {"command": command}, ok=False)
        return {"ok": False, "error": f"timed out after {timeout}s"}
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"shell not found: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def make_handler(name: str, **_kwargs):
    if name == "run_shell":
        return run_shell
    raise KeyError(name)
