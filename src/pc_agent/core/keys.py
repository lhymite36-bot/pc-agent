"""Multiple Gemini API keys with rotation on quota exhaustion."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from pc_agent.config import APP_ID, data_dir, get_api_key, set_api_key

logger = logging.getLogger("pc_agent.keys")
_lock = threading.Lock()


def _store_path() -> Path:
    return data_dir() / "api_keys.json"


def list_keys() -> list[dict]:
    """Return [{id, hint, cooling_until?}]. Never returns full secrets in 'hint' only."""
    path = _store_path()
    if not path.exists():
        primary = get_api_key()
        if primary:
            return [{"id": "primary", "hint": _hint(primary), "cooling_until": None}]
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return list(data.get("keys") or [])
    except Exception:
        return []


def _hint(key: str) -> str:
    k = (key or "").strip()
    if len(k) < 12:
        return "****"
    return k[:6] + "…" + k[-4:]


def _load_raw() -> dict:
    path = _store_path()
    if not path.exists():
        primary = get_api_key()
        raw = {"active": "primary", "secrets": {}, "meta": {}}
        if primary:
            raw["secrets"]["primary"] = primary
            raw["meta"]["primary"] = {"hint": _hint(primary), "cooling_until": None}
        return raw
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"active": "primary", "secrets": {}, "meta": {}}


def _save_raw(raw: dict) -> None:
    path = _store_path()
    # Persist secrets + meta; chmod best-effort
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    try:
        import os
        os.chmod(path, 0o600)
    except OSError:
        pass
    # Keep primary keyring entry in sync with active key
    active = raw.get("active")
    secret = (raw.get("secrets") or {}).get(active)
    if secret:
        try:
            set_api_key(secret)
        except Exception:
            pass


def add_key(key: str, key_id: str | None = None) -> dict:
    key = (key or "").strip()
    if not key:
        raise ValueError("empty key")
    with _lock:
        raw = _load_raw()
        kid = key_id or f"key{len(raw.get('secrets') or {}) + 1}"
        raw.setdefault("secrets", {})[kid] = key
        raw.setdefault("meta", {})[kid] = {"hint": _hint(key), "cooling_until": None}
        if not raw.get("active"):
            raw["active"] = kid
        _save_raw(raw)
        return {"id": kid, "hint": _hint(key)}


def active_key() -> str | None:
    with _lock:
        raw = _load_raw()
        secrets = raw.get("secrets") or {}
        active = raw.get("active")
        if active and active in secrets:
            # Skip if cooling
            meta = (raw.get("meta") or {}).get(active) or {}
            until = meta.get("cooling_until")
            if until:
                try:
                    if datetime.fromisoformat(until) > datetime.now(timezone.utc):
                        # try rotate
                        pass
                    else:
                        return secrets[active]
                except Exception:
                    return secrets[active]
            else:
                return secrets[active]
        # pick first non-cooling
        now = datetime.now(timezone.utc)
        for kid, secret in secrets.items():
            meta = (raw.get("meta") or {}).get(kid) or {}
            until = meta.get("cooling_until")
            if until:
                try:
                    if datetime.fromisoformat(until) > now:
                        continue
                except Exception:
                    pass
            raw["active"] = kid
            _save_raw(raw)
            return secret
        # fall back to keyring
        return get_api_key()


def mark_quota(key_id: str | None, retry_seconds: float) -> None:
    """Cool a key until now+retry_seconds and rotate to the next free key."""
    with _lock:
        raw = _load_raw()
        kid = key_id or raw.get("active") or "primary"
        until = datetime.now(timezone.utc).timestamp() + max(60.0, retry_seconds)
        iso = datetime.fromisoformat(
            datetime.now(timezone.utc).isoformat()
        )
        # store as iso
        from datetime import timedelta
        until_dt = datetime.now(timezone.utc) + timedelta(seconds=max(60.0, retry_seconds))
        raw.setdefault("meta", {}).setdefault(kid, {})["cooling_until"] = until_dt.isoformat()
        # rotate
        secrets = raw.get("secrets") or {}
        for other in secrets:
            if other == kid:
                continue
            meta = (raw.get("meta") or {}).get(other) or {}
            u = meta.get("cooling_until")
            if u:
                try:
                    if datetime.fromisoformat(u) > datetime.now(timezone.utc):
                        continue
                except Exception:
                    pass
            raw["active"] = other
            break
        _save_raw(raw)
        logger.info("API key %s cooling until %s; active=%s", kid, until_dt.isoformat(), raw.get("active"))


def list_status() -> list[dict]:
    with _lock:
        raw = _load_raw()
        out = []
        now = datetime.now(timezone.utc)
        for kid, secret in (raw.get("secrets") or {}).items():
            meta = (raw.get("meta") or {}).get(kid) or {}
            until = meta.get("cooling_until")
            cooling = False
            local_reset = None
            if until:
                try:
                    dt = datetime.fromisoformat(until)
                    if dt > now:
                        cooling = True
                        local_reset = dt.astimezone().strftime("%I:%M %p").lstrip("0")
                except Exception:
                    pass
            out.append(
                {
                    "id": kid,
                    "hint": meta.get("hint") or _hint(secret),
                    "active": kid == raw.get("active"),
                    "cooling": cooling,
                    "reset_local": local_reset,
                }
            )
        return out


def rotate() -> str | None:
    """Force rotate to next non-cooling key. Returns new key or None."""
    with _lock:
        raw = _load_raw()
        keys = list((raw.get("secrets") or {}).keys())
        if not keys:
            return get_api_key()
        try:
            idx = keys.index(raw.get("active"))
        except ValueError:
            idx = -1
        now = datetime.now(timezone.utc)
        for off in range(1, len(keys) + 1):
            kid = keys[(idx + off) % len(keys)]
            meta = (raw.get("meta") or {}).get(kid) or {}
            until = meta.get("cooling_until")
            if until:
                try:
                    if datetime.fromisoformat(until) > now:
                        continue
                except Exception:
                    pass
            raw["active"] = kid
            _save_raw(raw)
            return raw["secrets"][kid]
        return None
