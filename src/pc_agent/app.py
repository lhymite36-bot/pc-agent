"""PC Agent entrypoint — wires brain, memory, voice, tools, UI, tray, proactive."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import traceback
import webbrowser
from typing import Any

from pc_agent import __version__
from pc_agent.config import (
    CHAT_HOST,
    CHAT_PORT,
    get_api_key,
    load_config,
    save_config,
    set_api_key,
)
from pc_agent.core import safety
from pc_agent.core.autostart import disable_autostart, enable_autostart
from pc_agent.core.brain import Brain
from pc_agent.core.memory import MemoryStore
from pc_agent.core.planner import Planner
from pc_agent.core.scheduler import Scheduler
from pc_agent.core.screen_awareness import ScreenAwareness
from pc_agent.core.voice import VoiceEngine
from pc_agent.tools import build_tooling
from pc_agent.tools.screen import load_last_screenshot_bytes, take_screenshot
from pc_agent.ui.telegram_bot import TelegramBridge
from pc_agent.ui.tray import start_tray
from pc_agent.ui.web_ui import create_app, run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pc_agent")


class Agent:
    def __init__(self) -> None:
        self.memory = MemoryStore()
        self.cfg = load_config()
        self.api_key = get_api_key()
        self.voice = VoiceEngine(self.api_key)
        self.brain: Brain | None = None
        self.scheduler: Scheduler | None = None
        self.screen: ScreenAwareness | None = None
        self.planner: Planner | None = None
        self.tg: TelegramBridge | None = None
        self._confirm_events: dict[str, threading.Event] = {}
        self._confirm_results: dict[str, bool] = {}
        self._lock = threading.RLock()

    def bootstrap(self) -> None:
        if not self.api_key:
            logger.warning("No Gemini API key yet — open the UI to add one.")
        else:
            self._init_brain()

        # Confirm callback: log + auto-deny in headless; UI can override via event
        safety.set_confirm_callback(self._confirm)
        safety.start_kill_switch()

        if self.cfg.get("autostart", True):
            enable_autostart()

        declarations, handlers = build_tooling(
            self.memory, scheduler=None, planner=None
        )
        # Wire speak
        handlers["speak"] = lambda args: self.voice.speak(args.get("text") or "")

        self._handlers = handlers
        self._declarations = declarations

        if self.brain is not None:
            self.brain.tool_handlers = handlers
            self.brain.tool_declarations = declarations
            self.planner = Planner(
                self.brain,
                handlers,
                take_screenshot=lambda: take_screenshot({}),
            )
            # Rebuild skills handlers with planner
            declarations, handlers = build_tooling(
                self.memory, scheduler=None, planner=self.planner
            )
            handlers["speak"] = lambda args: self.voice.speak(args.get("text") or "")
            self._handlers = handlers
            self._declarations = declarations
            self.brain.tool_handlers = handlers
            self.brain.tool_declarations = declarations

        self.scheduler = Scheduler(
            self.memory,
            on_proactive=self._on_proactive,
            on_task=self._on_task,
        )
        # Rebind schedule tools with live scheduler
        declarations, handlers = build_tooling(
            self.memory, scheduler=self.scheduler, planner=self.planner
        )
        handlers["speak"] = lambda args: self.voice.speak(args.get("text") or "")
        self._handlers = handlers
        self._declarations = declarations
        if self.brain:
            self.brain.tool_handlers = handlers
            self.brain.tool_declarations = declarations

        self.scheduler.start()

        if self.brain and self.cfg.get("screen_awareness"):
            self.screen = ScreenAwareness(
                self.brain, take_screenshot=lambda: take_screenshot({})
            )
            self.brain.screen_summary = lambda: self.screen.last_summary if self.screen else ""
            self.screen.start()

        self.tg = TelegramBridge(self.chat)
        self.tg.start()

    def _init_brain(self) -> None:
        assert self.api_key
        self.brain = Brain(
            api_key=self.api_key,
            model=self.cfg.get("model"),
            memory_context_fn=self.memory.build_memory_context,
        )
        self.brain.pick_model(self.cfg.get("model"))
        self.memory.set_embedder(self.brain.embed)
        self.cfg["model"] = self.brain.model
        save_config(self.cfg)

    def _confirm(self, category: str, description: str) -> bool:
        """Default confirm: print and deny unless a UI event approves within 60s.

        The web UI currently uses the browser's confirm() for Memory deletes;
        for tool-time confirms we log loudly and deny in headless mode.
        """
        logger.warning("CONFIRM [%s]: %s", category, description)
        # If running with a TTY, ask there
        if sys.stdin and sys.stdin.isatty():
            try:
                ans = input(f"Allow {category}? {description} [y/N] ").strip().lower()
                return ans in {"y", "yes"}
            except Exception:
                return False
        return False

    def chat(self, text: str) -> str:
        with self._lock:
            if safety.is_halted():
                return "I'm halted (kill-switch). Resume from the tray or /api/resume."
            if not self.brain:
                key = get_api_key()
                if not key:
                    return "Add your Gemini API key in Settings first."
                self.api_key = key
                self.voice = VoiceEngine(key)
                self._init_brain()
                self.bootstrap_tools_only()

            # Persist user turn
            self.memory.add_message("user", text)

            # Maybe summarize old history
            if self.memory.needs_summarization() and self.brain:
                older = self.memory.messages_to_summarize()
                if older:
                    summary = self.brain.summarize(older)
                    if summary:
                        self.memory.add_summary(summary, older[-1]["id"])
                        self.memory.drop_messages_up_to(older[-1]["id"])

            # Attach last screenshot if user asks about on-screen content
            image = None
            lower = text.lower()
            if any(
                k in lower
                for k in ("on screen", "what's this", "whats this", "this error", "see this")
            ):
                if self.screen and self.screen.last_summary:
                    pass  # already in system prompt
                else:
                    take_screenshot({})
                    image = load_last_screenshot_bytes()

            assert self.brain is not None
            reply = self.brain.chat(text, image_bytes=image)
            self.memory.add_message("assistant", reply)
            self.memory.auto_extract(text, reply)

            if load_config().get("voice_enabled", True):
                # Speak a short form — first 2 sentences
                spoken = _shorten_for_speech(reply)
                try:
                    self.voice.speak(spoken)
                except Exception:
                    logger.exception("speak failed")
            return reply

    def bootstrap_tools_only(self) -> None:
        declarations, handlers = build_tooling(
            self.memory, scheduler=self.scheduler, planner=self.planner
        )
        handlers["speak"] = lambda args: self.voice.speak(args.get("text") or "")
        if self.brain:
            if self.planner is None:
                self.planner = Planner(
                    self.brain, handlers, take_screenshot=lambda: take_screenshot({})
                )
                declarations, handlers = build_tooling(
                    self.memory, scheduler=self.scheduler, planner=self.planner
                )
                handlers["speak"] = lambda args: self.voice.speak(args.get("text") or "")
            self.brain.tool_handlers = handlers
            self.brain.tool_declarations = declarations

    def _on_proactive(self, note: str) -> None:
        logger.info("Proactive: %s", note)
        self.memory.log_episode(title="proactive", detail=note, outcome="noticed")
        if load_config().get("voice_enabled", True):
            self.voice.speak(note)
        self.memory.add_message("assistant", f"[proactive] {note}")

    def _on_task(self, task: dict) -> None:
        import json

        payload = task.get("payload") or "{}"
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"prompt": payload}
        prompt = payload.get("prompt") or task.get("title") or ""
        if prompt:
            self.chat(f"[scheduled task] {prompt}")

    def update_settings(self, body: dict) -> None:
        cfg = load_config()
        for k in (
            "agent_name",
            "personality",
            "model",
            "proactive",
            "screen_awareness",
            "screen_awareness_interval_sec",
            "autostart",
            "confirm_dangerous",
            "telegram_token",
            "telegram_allowlist",
            "voice_mode",
            "voice_id",
            "voice_enabled",
            "browser_use_chrome_profile",
        ):
            if k in body:
                cfg[k] = body[k]
        save_config(cfg)
        self.cfg = cfg
        if cfg.get("autostart"):
            enable_autostart()
        else:
            disable_autostart()
        if cfg.get("screen_awareness") and self.brain:
            if self.screen is None:
                self.screen = ScreenAwareness(
                    self.brain, take_screenshot=lambda: take_screenshot({})
                )
                self.brain.screen_summary = lambda: self.screen.last_summary if self.screen else ""
                self.screen.start()
        elif self.screen and not cfg.get("screen_awareness"):
            self.screen.stop()
        if self.tg:
            self.tg.start()

    def run(self, open_browser: bool = True) -> None:
        self.bootstrap()
        app = create_app(
            chat_fn=self.chat,
            memory=self.memory,
            voice=self.voice,
            get_settings=load_config,
            update_settings=self.update_settings,
            set_key=self._set_key,
            halt_fn=lambda: safety.halt("ui"),
            resume_fn=safety.resume,
            status_fn=lambda: (self.brain.model_note if self.brain else ""),
        )
        run_server(app, CHAT_HOST, CHAT_PORT)
        url = f"http://{CHAT_HOST}:{CHAT_PORT}/"

        def open_ui() -> None:
            webbrowser.open(url)

        start_tray(
            open_ui=open_ui,
            halt=lambda: safety.halt("tray"),
            resume=safety.resume,
            quit_app=self.shutdown,
        )
        logger.info("PC Agent v%s listening at %s", __version__, url)
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        # Keep alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def _set_key(self, key: str) -> None:
        set_api_key(key)
        self.api_key = key
        self.voice = VoiceEngine(key)
        self._init_brain()
        self.bootstrap_tools_only()

    def shutdown(self) -> None:
        if self.scheduler:
            self.scheduler.stop()
        if self.screen:
            self.screen.stop()
        if self.tg:
            self.tg.stop()
        safety.stop_kill_switch()
        self.memory.close()
        sys.exit(0)


def _shorten_for_speech(text: str, max_chars: int = 400) -> str:
    text = text.strip()
    # Drop markdown fences / bullets noise lightly
    lines = [l for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)
    # First two sentences
    parts = []
    buf = ""
    for ch in joined:
        buf += ch
        if ch in ".!?" and len(buf) > 20:
            parts.append(buf.strip())
            buf = ""
            if len(parts) >= 2:
                break
    spoken = " ".join(parts) if parts else joined
    return spoken[:max_chars]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PC Agent — always-on desktop agent")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Restart the agent if it crashes (24/7 supervisor).",
    )
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return
    if args.watchdog:
        _watchdog()
        return
    Agent().run(open_browser=not args.no_browser)


def _watchdog() -> None:
    """Restart on crash for 24/7 operation."""
    import subprocess

    while True:
        logger.info("Watchdog starting agent…")
        proc = subprocess.Popen([sys.executable, "-m", "pc_agent", "--no-browser"])
        code = proc.wait()
        logger.error("Agent exited with %s; restarting in 3s", code)
        time.sleep(3)


if __name__ == "__main__":
    main()
