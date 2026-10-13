"""Model ranking + automatic failover across Gemini models.

Transient/unavailable errors (503, 429, 404, 500) -> one short retry, then the
next model in the ranked list. A working fallback is kept for a cooldown, after
which the preferred model is tried again.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Iterable

logger = logging.getLogger("pc_agent.models")

RETRYABLE = re.compile(
    r"\b(503|429|404|500)\b|UNAVAILABLE|RESOURCE_EXHAUSTED|NOT_FOUND|INTERNAL|high demand|overloaded|no longer available",
    re.I,
)
EXCLUDE_TEXT = ("lite", "tts", "embed", "image", "veo", "live", "audio", "imagen", "aqa", "gemma", "learnlm", "robotics", "computer-use")

LIST_TTL_SEC = 3 * 3600
COOLDOWN_SEC = 15 * 60
RETRY_DELAY_SEC = 1.0

ALL_FAILED_MSG = (
    "All the Gemini models I can reach are busy or unavailable right now. "
    "Give it a minute and try again — I'll keep checking."
)


class AllModelsFailed(RuntimeError):
    pass


def is_retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (404, 429, 500, 503):
        return True
    return bool(RETRYABLE.search(str(exc)))


def _version_key(name: str) -> tuple:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2) or 0) if m else 0
    stable = 0 if ("preview" in name or "exp" in name) else 1
    return (major, minor, stable)


def rank_text_models(names: Iterable[str]) -> list[str]:
    """Newest flash first, then pro, then other flash. Excludes lite/tts/etc."""
    good = []
    for n in names:
        n = n.replace("models/", "")
        low = n.lower()
        if not low.startswith("gemini-"):
            continue
        if any(x in low for x in EXCLUDE_TEXT):
            continue
        if "flash" not in low and "pro" not in low:
            continue
        good.append(n)
    good = list(dict.fromkeys(good))
    flash = sorted([n for n in good if "flash" in n], key=_version_key, reverse=True)
    pro = sorted([n for n in good if "pro" in n and "flash" not in n], key=_version_key, reverse=True)
    if not flash:
        return pro
    return [flash[0]] + pro + flash[1:]


def rank_embedding_models(names: Iterable[str]) -> list[str]:
    out = [n.replace("models/", "") for n in names if "embed" in n.lower()]
    out = list(dict.fromkeys(out))
    return sorted(out, key=lambda n: (("gemini" in n), _version_key(n), n), reverse=True)


class ModelRouter:
    """Holds a ranked candidate list per capability and routes calls with failover."""

    def __init__(
        self,
        kind: str,
        candidates: list[str],
        preferred: str | None = None,
        on_switch: Callable[[str, str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.kind = kind
        self._lock = threading.RLock()
        self.preferred = preferred
        self.candidates = self._order(candidates, preferred)
        self.current: str | None = self.candidates[0] if self.candidates else preferred
        self._fallback_since: float | None = None
        self.on_switch = on_switch
        self._sleep = sleep
        self._clock = clock
        self.last_note = ""

    @staticmethod
    def _order(cands: list[str], preferred: str | None) -> list[str]:
        c = list(dict.fromkeys(cands))
        if preferred:
            c = [preferred] + [x for x in c if x != preferred]
        return c

    def set_candidates(self, cands: list[str]) -> None:
        with self._lock:
            self.candidates = self._order(cands, self.preferred)
            if self.current not in self.candidates and self.candidates:
                self.current = self.candidates[0]

    def _start_order(self) -> list[str]:
        with self._lock:
            if (
                self._fallback_since is not None
                and self._clock() - self._fallback_since >= COOLDOWN_SEC
            ):
                # Cooldown over: try preferred again
                self._fallback_since = None
                self.current = self.candidates[0] if self.candidates else self.current
            cur = self.current
            if cur and cur in self.candidates:
                i = self.candidates.index(cur)
                return self.candidates[i:] + self.candidates[:i]
            return list(self.candidates)

    def call(self, fn: Callable[[str], Any]) -> Any:
        """fn(model_name) -> result. Raises AllModelsFailed if every model fails."""
        order = self._start_order()
        if not order:
            raise AllModelsFailed(f"no {self.kind} models")
        last: BaseException | None = None
        for model in order:
            for attempt in range(2):
                try:
                    result = fn(model)
                    self._mark_ok(model)
                    return result
                except Exception as exc:
                    last = exc
                    if not is_retryable(exc):
                        raise
                    logger.info("%s model %s failed (%s), attempt %d", self.kind, model, type(exc).__name__, attempt + 1)
                    if attempt == 0:
                        self._sleep(RETRY_DELAY_SEC)
        self.last_note = ALL_FAILED_MSG
        raise AllModelsFailed(str(last))

    def _mark_ok(self, model: str) -> None:
        with self._lock:
            prev = self.current
            self.current = model
            top = self.candidates[0] if self.candidates else model
            if model != top:
                if self._fallback_since is None or prev != model:
                    self._fallback_since = self._clock()
                self.last_note = f"Using {model}"
            else:
                self._fallback_since = None
                self.last_note = f"Using {model}"
            if prev != model and self.on_switch:
                try:
                    self.on_switch(prev or "", model)
                except Exception:
                    pass


class ModelCatalog:
    """Cached models.list → ranked lists for text and embeddings."""

    def __init__(self, client, clock: Callable[[], float] = time.time) -> None:
        self.client = client
        self._clock = clock
        self._at = 0.0
        self._names: list[str] = []
        self._lock = threading.Lock()

    def names(self, force: bool = False) -> list[str]:
        with self._lock:
            if not force and self._names and self._clock() - self._at < LIST_TTL_SEC:
                return self._names
            names: list[str] = []
            try:
                for m in self.client.models.list():
                    name = (getattr(m, "name", "") or "").replace("models/", "")
                    actions = str(
                        getattr(m, "supported_actions", None)
                        or getattr(m, "supported_generation_methods", None)
                        or ""
                    )
                    if actions and "embed" in name.lower():
                        if "embedContent" not in actions:
                            continue
                    elif actions and "generateContent" not in actions:
                        continue
                    names.append(name)
            except Exception as exc:
                logger.warning("models.list failed: %s", exc)
            if names:
                self._names = names
                self._at = self._clock()
            return self._names

    def text_models(self, fallback: list[str]) -> list[str]:
        ranked = rank_text_models(self.names())
        return ranked or list(fallback)

    def embedding_models(self, fallback: list[str]) -> list[str]:
        ranked = rank_embedding_models(self.names())
        return ranked or list(fallback)
