"""Optional Telegram control — replies only to allowlisted chat ids."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import httpx

from pc_agent.config import load_config
from pc_agent.core import safety

logger = logging.getLogger("pc_agent.telegram")


class TelegramBridge:
    def __init__(self, chat_fn: Callable[[str], str]) -> None:
        self.chat_fn = chat_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0

    def start(self) -> None:
        cfg = load_config()
        if not cfg.get("telegram_token"):
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pc-agent-tg", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            cfg = load_config()
            token = cfg.get("telegram_token") or ""
            allow = {str(x) for x in (cfg.get("telegram_allowlist") or [])}
            if not token or not allow:
                self._stop.wait(10)
                continue
            try:
                r = httpx.get(
                    f"https://api.telegram.org/bot{token}/getUpdates",
                    params={"offset": self._offset, "timeout": 20},
                    timeout=30,
                )
                data = r.json()
                for upd in data.get("result") or []:
                    self._offset = int(upd["update_id"]) + 1
                    msg = upd.get("message") or {}
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if not text or chat_id not in allow:
                        if text and chat_id not in allow:
                            logger.warning("Ignored Telegram chat_id=%s (not allowlisted)", chat_id)
                        continue
                    if safety.is_halted():
                        self._send(token, chat_id, "Agent is halted (kill-switch).")
                        continue
                    # Sending a message outward is confirm-gated when content looks like outreach
                    reply = self.chat_fn(text)
                    self._send(token, chat_id, reply[:3500] or "(no reply)")
            except Exception:
                logger.exception("telegram poll failed")
                self._stop.wait(5)

    def _send(self, token: str, chat_id: str, text: str) -> None:
        if not safety.confirm("send_message", f"Send Telegram reply to {chat_id}?"):
            # For allowlisted interactive chat, auto-approve short replies via callback;
            # if no UI, skip. Use a softer path: log and still send only if confirm disabled.
            cfg = load_config()
            if cfg.get("confirm_dangerous", True):
                # Avoid blocking the poll loop forever with no UI — send a notice instead.
                text = "[Confirmation required in PC Agent UI before Telegram replies. Open the app.]"
        try:
            httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=30,
            )
            safety.log_action("send_message", "telegram", {"chat_id": chat_id})
        except Exception:
            logger.exception("telegram send failed")
