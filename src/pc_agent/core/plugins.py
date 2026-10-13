"""Load custom Python tools from the user's plugins folder.

Each plugin is a .py file that defines:
  DECLARATIONS: list[dict]  (Gemini function declarations)
  HANDLERS: dict[str, callable]  mapping name -> (args: dict) -> dict
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

from pc_agent.config import plugins_dir

logger = logging.getLogger("pc_agent.plugins")


def load_plugins() -> tuple[list[dict], dict[str, Callable[[dict], Any]]]:
    declarations: list[dict] = []
    handlers: dict[str, Callable[[dict], Any]] = {}
    root = plugins_dir()
    # Drop a README once so the folder is discoverable
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Drop custom_tool.py files here.\n"
            "Each file must define DECLARATIONS (list) and HANDLERS (dict).\n",
            encoding="utf-8",
        )
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"pc_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            decls = getattr(mod, "DECLARATIONS", [])
            hands = getattr(mod, "HANDLERS", {})
            for d in decls:
                name = d.get("name")
                if not name or name in handlers:
                    logger.warning("plugin %s: skip duplicate/missing name", path.name)
                    continue
                if name not in hands:
                    logger.warning("plugin %s: no handler for %s", path.name, name)
                    continue
                declarations.append(d)
                handlers[name] = hands[name]
            logger.info("loaded plugin %s (%d tools)", path.name, len(decls))
        except Exception:
            logger.exception("failed to load plugin %s", path)
    return declarations, handlers
