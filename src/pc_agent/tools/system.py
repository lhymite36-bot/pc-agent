"""System info."""

from __future__ import annotations

import platform
import sys

DECLARATIONS = [
    {
        "name": "system_info",
        "description": "Get OS, CPU, memory, disk, and user info.",
        "parameters": {"type": "object", "properties": {}},
    }
]


def system_info(_args: dict) -> dict:
    try:
        import psutil
    except Exception:
        psutil = None
    info = {
        "ok": True,
        "platform": sys.platform,
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "hostname": platform.node(),
    }
    if psutil:
        vm = psutil.virtual_memory()
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_percent"] = psutil.cpu_percent(interval=0.2)
        info["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "percent": vm.percent,
        }
        try:
            disk = psutil.disk_usage("/")
            info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }
        except Exception:
            pass
    return info


def make_handler(name: str, **_kwargs):
    if name == "system_info":
        return system_info
    raise KeyError(name)
