"""Task queue runner + proactive watchers (disk, downloads, reminders)."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pc_agent.config import load_config
from pc_agent.core import safety

logger = logging.getLogger("pc_agent.scheduler")


class Scheduler:
    def __init__(
        self,
        memory,
        on_proactive: Callable[[str], None] | None = None,
        on_task: Callable[[dict], None] | None = None,
    ) -> None:
        self.memory = memory
        self.on_proactive = on_proactive
        self.on_task = on_task
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cron_jobs: dict[int, object] = {}
        self._aps = None
        self._last_disk_warn = 0.0
        self._seen_downloads: set[str] = set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._aps = BackgroundScheduler()
            self._aps.start()
        except Exception as exc:
            logger.warning("APScheduler unavailable: %s", exc)
            self._aps = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pc-agent-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._aps is not None:
            try:
                self._aps.shutdown(wait=False)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)

    def add_cron_task(self, task_id: int, cron: str, prompt: str) -> None:
        if self._aps is None:
            raise RuntimeError("APScheduler not available")
        from apscheduler.triggers.cron import CronTrigger

        parts = cron.split()
        if len(parts) != 5:
            raise ValueError("cron must be 5 fields: min hour dom month dow")
        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

        def _fire() -> None:
            if safety.is_halted():
                return
            self.memory.update_task(task_id, status="pending", run_at=_now())
            if self.on_task:
                self.on_task(
                    {
                        "id": task_id,
                        "title": f"cron:{task_id}",
                        "payload": json.dumps({"prompt": prompt}),
                        "status": "pending",
                    }
                )

        job = self._aps.add_job(_fire, trigger=trigger, id=f"task-{task_id}", replace_existing=True)
        self._cron_jobs[task_id] = job

    def cancel(self, task_id: int) -> None:
        job = self._cron_jobs.pop(task_id, None)
        if job is not None and self._aps is not None:
            try:
                self._aps.remove_job(f"task-{task_id}")
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not safety.is_halted():
                    self._run_due_tasks()
                    if load_config().get("proactive", True):
                        self._proactive_tick()
            except Exception:
                logger.exception("scheduler tick failed")
            self._stop.wait(15)

    def _run_due_tasks(self) -> None:
        for task in self.memory.due_tasks():
            self.memory.update_task(task["id"], status="running")
            try:
                if self.on_task:
                    self.on_task(task)
                self.memory.update_task(task["id"], status="done")
                self.memory.log_episode(
                    title=task.get("title") or f"task {task['id']}",
                    detail=task.get("payload", ""),
                    outcome="done",
                )
            except Exception as exc:
                self.memory.update_task(task["id"], status="failed", last_error=str(exc))

    def _proactive_tick(self) -> None:
        """Notice low disk, failed downloads, due reminders — speak up via callback."""
        notes: list[str] = []
        # Low disk
        try:
            usage = shutil.disk_usage(str(Path.home()))
            free_gb = usage.free / (1024**3)
            if free_gb < 5 and time.time() - self._last_disk_warn > 3600:
                notes.append(f"Heads up — only {free_gb:.1f} GB free on the system drive.")
                self._last_disk_warn = time.time()
        except Exception:
            pass

        # Failed / stuck downloads (Windows + common folders)
        try:
            dl = Path.home() / "Downloads"
            if dl.is_dir():
                for p in dl.glob("*.crdownload"):
                    key = str(p)
                    if key not in self._seen_downloads and p.stat().st_mtime < time.time() - 600:
                        notes.append(f"A Chrome download looks stuck: {p.name}")
                        self._seen_downloads.add(key)
                for p in dl.glob("*.tmp"):
                    key = str(p)
                    if key not in self._seen_downloads and p.stat().st_mtime < time.time() - 600:
                        notes.append(f"A download temp file has been sitting there: {p.name}")
                        self._seen_downloads.add(key)
        except Exception:
            pass

        # Reminder tasks stored in memory with run_at
        try:
            for task in self.memory.due_tasks():
                title = task.get("title") or ""
                if title.lower().startswith("remind"):
                    notes.append(f"Reminder: {title}")
        except Exception:
            pass

        if notes and self.on_proactive:
            for n in notes:
                try:
                    self.on_proactive(n)
                except Exception:
                    logger.exception("proactive callback failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
