"""Model ranking + fast automatic failover across Gemini models.

Policy (latency first):
- 404/NOT_FOUND  -> mark model dead (persisted), skip immediately, no retry.
- 429/RESOURCE_EXHAUSTED -> cooldown honoring retryDelay (persisted), skip, no retry.
- 503/UNAVAILABLE -> at most one quick retry (300 ms), then next model.
- 500/INTERNAL -> next model, no retry.
- Network errors (RemoteProtocolError, ReadTimeout, ConnectError, …) -> cooldown, failover.
- First-token / stream timeouts -> cooldown (slow), failover.
- Rank by measured first-token latency (fast healthy models first); newest is a tie-break.
- models.list is never fetched in the request path (background refresh only).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger("pc_agent.models")

EXCLUDE_TEXT = (
    "lite", "tts", "embed", "image", "veo", "live", "audio", "imagen", "aqa",
    "gemma", "learnlm", "robotics", "computer-use", "customtools", "thinking-exp",
)

LIST_TTL_SEC = 3 * 3600
COOLDOWN_SEC = 15 * 60
RETRY_503_DELAY_SEC = 0.3
DEFAULT_429_COOLDOWN_SEC = 60.0
DEAD_TTL_SEC = 24 * 3600
NETWORK_COOLDOWN_SEC = 60.0
SLOW_COOLDOWN_SEC = 10 * 60  # model that timed out / was very slow
UNKNOWN_LATENCY = 9_999.0
LATENCY_WINDOW = 12

# Synthetic codes for our own failures
CODE_NETWORK = 598
CODE_TIMEOUT = 599

FIRST_TOKEN_TIMEOUT_SEC = 10.0
STREAM_TOTAL_TIMEOUT_SEC = 60.0
SLOW_FIRST_TOKEN_SEC = 8.0  # successful but slow → demote + short cool

ALL_FAILED_MSG = (
    "All the Gemini models I can reach are busy or unavailable right now. "
    "Give it a minute and try again — I'll keep checking."
)

FRIENDLY_TIMEOUT_MSG = (
    "That took too long, so I switched models. "
    "If this keeps happening, try again in a moment."
)

NETWORK_EXC_NAMES = {
    "RemoteProtocolError",
    "ReadTimeout",
    "ConnectError",
    "ConnectTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "TimeoutException",
    "NetworkError",
    "RemoteDisconnected",
    "ProtocolError",
    "ReadError",
    "WriteError",
    "LocalProtocolError",
}


class AllModelsFailed(RuntimeError):
    pass


class FirstTokenTimeout(TimeoutError):
    """No first stream chunk within FIRST_TOKEN_TIMEOUT_SEC."""


class StreamTimeout(TimeoutError):
    """Stream exceeded STREAM_TOTAL_TIMEOUT_SEC."""


def _code(exc: BaseException) -> int | None:
    if isinstance(exc, FirstTokenTimeout) or isinstance(exc, StreamTimeout):
        return CODE_TIMEOUT
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    name = type(exc).__name__
    if name in NETWORK_EXC_NAMES or any(n in name for n in NETWORK_EXC_NAMES):
        return CODE_NETWORK
    # Walk causes (httpx wraps)
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        c = _code(cause)
        if c:
            return c
    s = str(exc)
    if any(n in s for n in NETWORK_EXC_NAMES) or "Server disconnected" in s:
        return CODE_NETWORK
    m = re.search(r"\b(404|429|500|503)\b", s)
    if m:
        return int(m.group(1))
    if "NOT_FOUND" in s or "no longer available" in s:
        return 404
    if "RESOURCE_EXHAUSTED" in s:
        return 429
    if "UNAVAILABLE" in s or "high demand" in s or "overloaded" in s:
        return 503
    if "INTERNAL" in s:
        return 500
    if "timed out" in s.lower() or "timeout" in s.lower():
        return CODE_TIMEOUT
    return None


def is_retryable(exc: BaseException) -> bool:
    return _code(exc) in (404, 429, 500, 503, CODE_NETWORK, CODE_TIMEOUT)


def retry_delay_sec(exc: BaseException) -> float:
    """Parse retryDelay / 'retry in 7h26m' from a 429 error."""
    try:
        from pc_agent.core.quota import parse_retry_seconds
        secs = parse_retry_seconds(exc)
        if secs is not None:
            return float(secs)
    except Exception:
        pass
    s = str(exc)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s) or re.search(
        r"retry in (\d+(?:\.\d+)?)\s*s", s, re.I
    )
    return float(m.group(1)) if m else DEFAULT_429_COOLDOWN_SEC


def _version_key(name: str) -> tuple:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2) or 0) if m else 0
    stable = 0 if ("preview" in name or "exp" in name) else 1
    latest_alias = 0 if name.endswith("-latest") else 1
    return (major, minor, stable, latest_alias)


def rank_text_models(
    names: Iterable[str],
    latency: dict[str, float] | None = None,
) -> list[str]:
    """Flash first (by measured latency, then newest), then pro. Latency wins over 'newest'."""
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
    lat = latency or {}

    def sort_key(n: str) -> tuple:
        # Lower latency first; unknown near the back of the measured pack but still ahead of pro
        L = lat.get(n, UNKNOWN_LATENCY)
        # Prefer having a measurement slightly; unknowns sorted by version among themselves
        known = 0 if n in lat else 1
        ver = _version_key(n)
        # Negate version so newer wins ties
        return (known, L, (-ver[0], -ver[1], -ver[2], -ver[3]))

    flash = sorted([n for n in good if "flash" in n], key=sort_key)
    pro = sorted([n for n in good if "flash" not in n], key=sort_key)
    return flash + pro


def rank_embedding_models(names: Iterable[str]) -> list[str]:
    out = [n.replace("models/", "") for n in names if "embed" in n.lower()]
    out = list(dict.fromkeys(out))
    return sorted(out, key=lambda n: (("gemini" in n), _version_key(n), n), reverse=True)


class ModelHealth:
    """Dead/cooldown + rolling first-token latency, persisted across restarts."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self.dead: dict[str, float] = {}
        self.cooling: dict[str, float] = {}
        self.latency: dict[str, list[float]] = {}  # model -> recent first-token secs
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            now = self._clock()
            self.dead = {k: v for k, v in (data.get("dead") or {}).items() if now - v < DEAD_TTL_SEC}
            self.cooling = {k: v for k, v in (data.get("cooling") or {}).items() if v > now}
            raw = data.get("latency") or {}
            self.latency = {k: list(v)[-LATENCY_WINDOW:] for k, v in raw.items() if isinstance(v, list)}
        except Exception as exc:
            logger.debug("model health load failed: %s", exc)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"dead": self.dead, "cooling": self.cooling, "latency": self.latency}),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("model health save failed: %s", exc)

    def usable(self, model: str) -> bool:
        with self._lock:
            if model in self.dead:
                return False
            until = self.cooling.get(model)
            if until and until > self._clock():
                return False
            if until:
                self.cooling.pop(model, None)
            return True

    def mark_dead(self, model: str) -> None:
        with self._lock:
            self.dead[model] = self._clock()
            self._save()

    def mark_cooling(self, model: str, seconds: float) -> None:
        with self._lock:
            until = self._clock() + max(1.0, seconds)
            prev = self.cooling.get(model, 0)
            self.cooling[model] = max(prev, until)
            self._save()

    def record_latency(self, model: str, seconds: float) -> None:
        with self._lock:
            arr = self.latency.setdefault(model, [])
            arr.append(float(seconds))
            del arr[:-LATENCY_WINDOW]
            self._save()

    def avg_latency(self, model: str) -> float | None:
        with self._lock:
            arr = self.latency.get(model) or []
            if not arr:
                return None
            return sum(arr) / len(arr)

    def latency_map(self) -> dict[str, float]:
        with self._lock:
            return {k: sum(v) / len(v) for k, v in self.latency.items() if v}

    def clear(self, model: str) -> None:
        with self._lock:
            changed = self.dead.pop(model, None) is not None or self.cooling.pop(model, None) is not None
            if changed:
                self._save()


class ModelRouter:
    """Ranked candidates + fast failover. fn(model) does the API call."""

    def __init__(
        self,
        kind: str,
        candidates: list[str],
        preferred: str | None = None,
        on_switch: Callable[[str, str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        health: ModelHealth | None = None,
    ) -> None:
        self.kind = kind
        self._lock = threading.RLock()
        self.preferred = preferred
        self.candidates = list(dict.fromkeys(candidates))
        self.current: str | None = self.candidates[0] if self.candidates else preferred
        self._fallback_since: float | None = None
        self.on_switch = on_switch
        self._sleep = sleep
        self._clock = clock
        self.health = health or ModelHealth(None, clock=clock)
        self.last_note = ""
        self.last_attempts: list[tuple[str, str, float]] = []

    def set_candidates(self, cands: list[str]) -> None:
        """Replace candidate list. Order is respected (already latency-ranked)."""
        with self._lock:
            self.candidates = list(dict.fromkeys(cands))
            if self.current not in self.candidates and self.candidates:
                self.current = self.candidates[0]

    def _start_order(self) -> list[str]:
        with self._lock:
            if self._fallback_since is not None and self._clock() - self._fallback_since >= COOLDOWN_SEC:
                self._fallback_since = None
                # Retry preferred (or best ranked) after sticky cooldown
                if self.preferred and self.health.usable(self.preferred):
                    self.current = self.preferred
                else:
                    self.current = next(
                        (m for m in self.candidates if self.health.usable(m)), self.current
                    )
            cur = self.current
            order = list(self.candidates)
            if cur and cur in order:
                order = [cur] + [m for m in order if m != cur]
        return [m for m in order if self.health.usable(m)]

    def call(self, fn: Callable[[str], Any]) -> Any:
        order = self._start_order()
        self.last_attempts = []
        if not order:
            self.last_note = ALL_FAILED_MSG
            raise AllModelsFailed(f"no usable {self.kind} models")
        last: BaseException | None = None
        for model in order:
            tries = 0
            while True:
                t0 = self._clock()
                try:
                    result = fn(model)
                    dt = self._clock() - t0
                    self.last_attempts.append((model, "ok", dt))
                    self.health.record_latency(model, dt)
                    if dt >= SLOW_FIRST_TOKEN_SEC:
                        # Successful but too slow — demote for a while
                        self.health.mark_cooling(model, SLOW_COOLDOWN_SEC)
                        logger.info("%s model %s slow (%.1fs) — cooling", self.kind, model, dt)
                    self._mark_ok(model)
                    return result
                except Exception as exc:
                    code = _code(exc)
                    dt = self._clock() - t0
                    last = exc
                    if code is None:
                        self.last_attempts.append((model, f"error:{type(exc).__name__}", dt))
                        raise
                    self.last_attempts.append((model, str(code), dt))
                    # Record a penalty latency so ranking avoids this model
                    self.health.record_latency(model, max(dt, FIRST_TOKEN_TIMEOUT_SEC))
                    if code == 404:
                        self.health.mark_dead(model)
                        break
                    if code == 429:
                        self.health.mark_cooling(model, retry_delay_sec(exc))
                        break
                    if code == CODE_NETWORK:
                        self.health.mark_cooling(model, NETWORK_COOLDOWN_SEC)
                        break
                    if code == CODE_TIMEOUT:
                        self.health.mark_cooling(model, SLOW_COOLDOWN_SEC)
                        break
                    if code == 503 and tries == 0:
                        tries += 1
                        self._sleep(RETRY_503_DELAY_SEC)
                        continue
                    if code in (500, 503):
                        self.health.mark_cooling(model, 30)
                    break
        self.last_note = ALL_FAILED_MSG
        raise AllModelsFailed(str(last))

    def _mark_ok(self, model: str) -> None:
        with self._lock:
            prev = self.current
            self.current = model
            home = self.preferred if self.preferred else (
                self.candidates[0] if self.candidates else model
            )
            if model != home:
                if self._fallback_since is None:
                    self._fallback_since = self._clock()
            else:
                self._fallback_since = None
            self.last_note = f"Using {model}"
            if prev != model and self.on_switch:
                try:
                    self.on_switch(prev or "", model)
                except Exception:
                    pass


class ModelCatalog:
    """models.list cache. Request path only ever reads the cache."""

    def __init__(self, client, clock: Callable[[], float] = time.time, cache_path: Path | None = None) -> None:
        self.client = client
        self._clock = clock
        self._at = 0.0
        self._names: list[str] = []
        self._lock = threading.Lock()
        self.cache_path = cache_path
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
            self._names = list(data.get("names") or [])
            self._at = float(data.get("at") or 0)
        except Exception:
            pass

    def stale(self) -> bool:
        return not self._names or self._clock() - self._at >= LIST_TTL_SEC

    def refresh(self) -> list[str]:
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
            with self._lock:
                self._names = names
                self._at = self._clock()
            if self.cache_path:
                try:
                    self.cache_path.write_text(json.dumps({"names": names, "at": self._at}), encoding="utf-8")
                except Exception:
                    pass
        return self._names

    def refresh_async(self, on_done: Callable[[], None] | None = None) -> threading.Thread:
        def _run() -> None:
            self.refresh()
            if on_done:
                try:
                    on_done()
                except Exception:
                    logger.exception("catalog on_done failed")
        t = threading.Thread(target=_run, name="pc-agent-models", daemon=True)
        t.start()
        return t

    def names(self, force: bool = False) -> list[str]:
        if force:
            return self.refresh()
        return self._names

    def text_models(self, fallback: list[str], latency: dict[str, float] | None = None) -> list[str]:
        ranked = rank_text_models(self.names(), latency=latency)
        return ranked or rank_text_models(fallback, latency=latency) or list(fallback)

    def embedding_models(self, fallback: list[str]) -> list[str]:
        ranked = rank_embedding_models(self.names())
        return ranked or list(fallback)
