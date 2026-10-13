"""Gemini brain: streaming, low-thinking chat with fast model failover."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterator

from pc_agent.config import (
    DEFAULT_MODEL,
    EMBEDDING_FALLBACKS,
    EMBEDDING_MODEL,
    FALLBACK_MODELS,
    VERSION,
    data_dir,
    load_config,
    save_config,
)
from pc_agent.core.models import (
    ALL_FAILED_MSG,
    LIST_TTL_SEC,
    AllModelsFailed,
    ModelCatalog,
    ModelHealth,
    ModelRouter,
)

logger = logging.getLogger("pc_agent.brain")
timing = logging.getLogger("pc_agent.timing")

MAX_HISTORY_CONTENTS = 24   # capped window sent to the model
MAX_TOOL_ROUNDS = 10
MAX_TOOL_RESULT_CHARS = 6000


def build_system_prompt(agent_name: str, personality: str) -> str:
    """Compact system prompt — every token here is paid on every round."""
    return (
        f"You are {agent_name}, PC Agent v{VERSION}: an always-on assistant controlling this Windows PC via tools.\n"
        f"Personality: {personality}\n"
        "Be brief and act. Use tools instead of describing steps.\n"
        "Memory: call remember for lasting facts/preferences, recall when past context matters, "
        "forget on request, log_episode after multi-step jobs.\n"
        "Use run_plan for 3+ step goals; save_skill for repeated workflows.\n"
        "Safety: deletes, installs, sending messages/emails, payments, registry, shutdown and Chrome-profile "
        "use go through the confirmation flow; if halted, stop and say so.\n"
        "Replies may be spoken: short, casual sentences with contractions; no stage directions."
    )


def thinking_config(types, model: str, level: str = "minimal"):
    """Low thinking for chat, higher for planning. Returns None if unsupported."""
    low = model.lower()
    try:
        if low.startswith("gemini-3"):
            lvl = {"minimal": "minimal", "low": "low", "high": "high"}.get(level, "low")
            return types.ThinkingConfig(thinking_level=lvl)
        if "2.5" in low:
            budget = {"minimal": 0 if "flash" in low else 128, "low": 512, "high": 4096}.get(level, 0)
            return types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        return None
    return None


def _trim_history(history: list[Any], limit: int = MAX_HISTORY_CONTENTS) -> list[Any]:
    """Keep the last `limit` contents, starting at a plain user-text turn
    so function_call / function_response pairs are never split."""
    if len(history) <= limit:
        return history
    tail = history[-limit:]
    for i, c in enumerate(tail):
        parts = getattr(c, "parts", None) or []
        if getattr(c, "role", "") == "user" and any(getattr(p, "text", None) for p in parts) and not any(
            getattr(p, "function_response", None) for p in parts
        ):
            return tail[i:]
    return tail[-1:]


class Brain:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        tool_handlers: dict[str, Callable[[dict], Any]] | None = None,
        tool_declarations: list[dict] | None = None,
        memory_context_fn: Callable[[str], str] | None = None,
        start_background: bool = True,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        self.tool_handlers = tool_handlers or {}
        self.tool_declarations = tool_declarations or []
        self.memory_context_fn = memory_context_fn
        self._history: list[Any] = []
        self._tools_cache: tuple[int, Any] | None = None
        self._persist_switch = False
        self.health = ModelHealth(data_dir() / "model_health.json")
        self.catalog = ModelCatalog(self.client, cache_path=data_dir() / "models_cache.json")
        self.router = ModelRouter("text", list(FALLBACK_MODELS), preferred=self.model,
                                  on_switch=self._on_switch, health=self.health)
        self.embed_router = ModelRouter("embed", list(EMBEDDING_FALLBACKS),
                                        preferred=EMBEDDING_MODEL, health=self.health)
        self._apply_catalog()
        if start_background:
            self._start_catalog_refresher()

    # --- model management (never blocks the request path) ---

    def _apply_catalog(self) -> None:
        self.router.set_candidates(self.catalog.text_models(FALLBACK_MODELS))
        self.embed_router.set_candidates(self.catalog.embedding_models(EMBEDDING_FALLBACKS))

    def _start_catalog_refresher(self) -> None:
        def _loop() -> None:
            while True:
                if self.catalog.stale():
                    self.catalog.refresh()
                    self._apply_catalog()
                time.sleep(600)
        threading.Thread(target=_loop, name="pc-agent-catalog", daemon=True).start()

    def refresh_models(self, force: bool = False) -> None:
        """Re-apply cached catalog. force=True blocks on models.list (background use only)."""
        if force:
            self.catalog.refresh()
        self._apply_catalog()

    def _on_switch(self, old: str, new: str) -> None:
        self.model = new
        logger.info("text model switched %s -> %s", old, new)
        if self._persist_switch:
            cfg = load_config()
            cfg["model"] = new
            save_config(cfg)
            self._persist_switch = False

    @property
    def model_note(self) -> str:
        return self.router.last_note or f"Using {self.router.current or self.model}"

    def list_models(self) -> list[str]:
        return self.catalog.text_models(FALLBACK_MODELS)

    def pick_model(self, preferred: str | None = None) -> str:
        ranked = self.catalog.text_models(FALLBACK_MODELS)
        if preferred and (preferred in ranked or not self.catalog.names()):
            self.model = preferred
        elif ranked:
            self.model = ranked[0]
        else:
            self.model = preferred or DEFAULT_MODEL
        self.router.preferred = self.model
        self.router.set_candidates(ranked)
        self.router.current = self.model
        return self.model

    def _log_attempts(self, stage: str, t0: float) -> None:
        tries = ", ".join(f"{m}={o}({s:.2f}s)" for m, o, s in self.router.last_attempts)
        timing.info("%s %.2fs [%s]", stage, time.time() - t0, tries)

    def _note_404(self, model: str, exc: Exception) -> None:
        s = str(exc)
        if ("404" in s or "NOT_FOUND" in s) and model == self.router.preferred:
            self._persist_switch = True

    def generate(self, contents, config=None, thinking: str = "minimal"):
        """Non-streaming generate with failover. Raises AllModelsFailed."""
        types = self._types
        t0 = time.time()

        def _call(model):
            cfg = config or types.GenerateContentConfig()
            cfg = self._with_thinking(cfg, model, thinking)
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=cfg)
            except Exception as exc:
                if self._thinking_rejected(exc) and getattr(cfg, "thinking_config", None) is not None:
                    cfg.thinking_config = None
                    return self.client.models.generate_content(model=model, contents=contents, config=cfg)
                self._note_404(model, exc)
                raise

        try:
            return self.router.call(_call)
        finally:
            self._log_attempts("generate", t0)

    def _with_thinking(self, cfg, model: str, level: str):
        try:
            tc = thinking_config(self._types, model, level)
            cfg = cfg.model_copy() if hasattr(cfg, "model_copy") else cfg
            cfg.thinking_config = tc
        except Exception:
            pass
        return cfg

    @staticmethod
    def _thinking_rejected(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("400" in s or "invalid" in s) and "think" in s

    def generate_stream(self, contents, config, thinking: str = "minimal") -> Iterator[Any]:
        """Streaming generate with failover. Failover happens before the first chunk is yielded."""
        t0 = time.time()

        def _open(model):
            cfg = self._with_thinking(config, model, thinking)
            try:
                it = iter(self.client.models.generate_content_stream(model=model, contents=contents, config=cfg))
                first = next(it)  # errors surface here → router can fail over
            except StopIteration:
                return model, None, iter(())
            except Exception as exc:
                if self._thinking_rejected(exc) and getattr(cfg, "thinking_config", None) is not None:
                    cfg.thinking_config = None
                    it = iter(self.client.models.generate_content_stream(model=model, contents=contents, config=cfg))
                    first = next(it)
                else:
                    self._note_404(model, exc)
                    raise
            return model, first, it

        model, first, rest = self.router.call(_open)
        self._log_attempts("first_chunk", t0)
        if first is not None:
            yield first
        for chunk in rest:
            yield chunk
        timing.info("stream_done %.2fs model=%s", time.time() - t0, model)

    # --- embeddings ---

    def embed(self, text: str) -> list[float] | None:
        try:
            result = self.embed_router.call(
                lambda m: self.client.models.embed_content(model=m, contents=text)
            )
            emb = getattr(result, "embeddings", None) or getattr(result, "embedding", None)
            if emb is None:
                return None
            if isinstance(emb, list):
                values = getattr(emb[0], "values", None) if emb else None
            else:
                values = getattr(emb, "values", None)
            return list(values) if values else None
        except Exception as exc:
            logger.debug("embed failed: %s", exc)
            return None

    # --- chat ---

    def _system_instruction(self, user_text: str) -> str:
        cfg = load_config()
        base = build_system_prompt(cfg.get("agent_name") or "Atlas", cfg.get("personality") or "Witty, warm, and sharp.")
        if self.memory_context_fn:
            t0 = time.time()
            ctx = self.memory_context_fn(user_text) or ""
            timing.info("memory_context %.3fs", time.time() - t0)
            if ctx.strip():
                base += "\n\n" + ctx
        screen_summary = getattr(self, "screen_summary", None)
        if callable(screen_summary):
            try:
                s = screen_summary() or ""
            except Exception:
                s = ""
            if s.strip():
                base += "\n\n## Latest screen\n" + s.strip()
        return base

    def _tools_config(self):
        types = self._types
        if not self.tool_declarations:
            return None
        key = id(self.tool_declarations)
        if self._tools_cache and self._tools_cache[0] == key:
            return self._tools_cache[1]
        decls = [
            types.FunctionDeclaration(name=d["name"], description=d.get("description", ""), parameters=d.get("parameters"))
            for d in self.tool_declarations
        ]
        tools = [types.Tool(function_declarations=decls)]
        self._tools_cache = (key, tools)
        return tools

    def chat(self, user_text: str, image_bytes: bytes | None = None) -> str:
        return "".join(self.chat_stream(user_text, image_bytes=image_bytes))

    def chat_stream(self, user_text: str, image_bytes: bytes | None = None, thinking: str = "minimal") -> Iterator[str]:
        """Yield reply text deltas as they arrive; runs the tool loop in between."""
        types = self._types
        t_start = time.time()
        parts: list[Any] = [types.Part.from_text(text=user_text)]
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        user_msg = types.Content(role="user", parts=parts)
        self._history = _trim_history(self._history + [user_msg])

        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction(user_text),
            tools=self._tools_config(),
            temperature=0.4,
        )
        timing.info("prep %.3fs", time.time() - t_start)

        for rnd in range(MAX_TOOL_ROUNDS):
            text_parts: list[str] = []
            model_parts: list[Any] = []
            fn_calls: list[Any] = []
            try:
                for chunk in self.generate_stream(self._history, config, thinking=thinking):
                    cand = chunk.candidates[0] if getattr(chunk, "candidates", None) else None
                    if not cand or not cand.content:
                        continue
                    for p in cand.content.parts or []:
                        if getattr(p, "thought", False):
                            continue
                        if getattr(p, "function_call", None):
                            fn_calls.append(p.function_call)
                            model_parts.append(p)
                        elif getattr(p, "text", None):
                            text_parts.append(p.text)
                            model_parts.append(p)
                            yield p.text
            except AllModelsFailed:
                if self._history and self._history[-1] is user_msg:
                    self._history.pop()
                yield ALL_FAILED_MSG
                return
            except Exception:
                logger.exception("generate failed")
                yield "Something went wrong talking to the model. Try that again in a moment."
                return

            if model_parts:
                self._history.append(types.Content(role="model", parts=model_parts))
            if not fn_calls:
                if not text_parts:
                    yield "(no text)"
                timing.info("chat_total %.2fs rounds=%d", time.time() - t_start, rnd + 1)
                return

            result_parts = []
            for fc in fn_calls:
                t_tool = time.time()
                name, args = fc.name, dict(fc.args or {})
                handler = self.tool_handlers.get(name)
                if handler is None:
                    result = {"ok": False, "error": f"unknown tool: {name}"}
                else:
                    try:
                        result = handler(args)
                    except Exception as exc:
                        logger.exception("tool %s failed", name)
                        result = {"ok": False, "error": str(exc)}
                if not isinstance(result, dict):
                    result = {"result": result}
                result = _compact_result(result)
                timing.info("tool %s %.2fs", name, time.time() - t_tool)
                result_parts.append(types.Part.from_function_response(name=name, response=result))
            self._history.append(types.Content(role="user", parts=result_parts))
            if text_parts:
                yield "\n"

        yield "Stopped after too many tool rounds. Try a simpler request."

    def summarize(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in messages)
        prompt = (
            "Summarize this conversation for future context. Keep durable facts, preferences, decisions, "
            "and open tasks. Under 250 words.\n\n" + transcript
        )
        try:
            response = self.generate(prompt, self._types.GenerateContentConfig(temperature=0.2))
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning("summarize failed: %s", exc)
            return transcript[:800]

    def reset_session(self) -> None:
        self._history.clear()


def _compact_result(result: dict) -> dict:
    """Trim huge tool outputs (e.g. base64 previews) to keep rounds fast."""
    out = {}
    for k, v in result.items():
        if k == "preview_png_base64":
            continue
        if isinstance(v, str) and len(v) > MAX_TOOL_RESULT_CHARS:
            v = v[:MAX_TOOL_RESULT_CHARS] + "…[truncated]"
        out[k] = v
    return out
