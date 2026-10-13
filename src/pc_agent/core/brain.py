"""Gemini brain with function calling."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pc_agent.config import DEFAULT_MODEL, EMBEDDING_MODEL, EMBEDDING_FALLBACKS, FALLBACK_MODELS, VERSION, load_config, save_config
from pc_agent.core.models import ALL_FAILED_MSG, AllModelsFailed, ModelCatalog, ModelRouter

logger = logging.getLogger("pc_agent.brain")

def build_system_prompt(agent_name: str, personality: str) -> str:
    return f"""You are {agent_name}, a PC Agent (v{VERSION}) — an always-on desktop assistant that controls this Windows PC.
You are designed to be *better than a chatbox*: you notice things, plan multi-step work, verify results, and remember.

Personality: {personality}

Capabilities: shell, files, apps/URLs, screenshots+vision, mouse/keyboard, windows, clipboard,
system info, scheduling, browser automation (Playwright), skills (reusable workflows), plugins,
long-term memory, and a multi-step planner with screenshot self-check (`run_plan`).

Voice: when speaking aloud, keep lines short and conversational (contractions, casual). The voice
engine already styles delivery — don't write stage directions.

Memory rules (critical):
- When the user shares a lasting fact or preference, call `remember`.
- Before acting on something that may have prior context, call `recall`.
- When the user asks you to forget something, call `forget`.
- After finishing a multi-step job, call `log_episode` with the outcome.
- Relevant long-term memories and a conversation summary are injected below when available — use them.

For non-trivial goals (3+ steps), prefer `run_plan` so you verify with a screenshot and retry.
Save repeated workflows with `save_skill`.

Safety rules:
- Never delete files, install software, send messages/emails, make payments, edit the registry,
  shut down/reboot, or use the user's Chrome profile without the confirmation flow.
- Prefer reversible actions. Explain risky steps.
- If the kill-switch is active, stop and tell the user.

Be concise, practical, and proactive. Speak as the user's agent on their PC.
"""


class Brain:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        tool_handlers: dict[str, Callable[[dict], Any]] | None = None,
        tool_declarations: list[dict] | None = None,
        memory_context_fn: Callable[[str], str] | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        self.tool_handlers = tool_handlers or {}
        self.tool_declarations = tool_declarations or []
        self.memory_context_fn = memory_context_fn
        self._history: list[Any] = []  # in-session Gemini contents
        self.catalog = ModelCatalog(self.client)
        self.router = ModelRouter("text", list(FALLBACK_MODELS), preferred=self.model,
                                  on_switch=self._on_switch)
        self.embed_router = ModelRouter("embed", list(EMBEDDING_FALLBACKS), preferred=EMBEDDING_MODEL)

    def _on_switch(self, old: str, new: str) -> None:
        self.model = new
        logger.info("text model switched %s -> %s", old, new)
        # Persist only when the configured model is gone (404); otherwise keep preferred
        cfg = load_config()
        if getattr(self, "_persist_switch", False):
            cfg["model"] = new
            save_config(cfg)
            self._persist_switch = False

    @property
    def model_note(self) -> str:
        return self.router.last_note or f"Using {self.router.current or self.model}"

    def refresh_models(self, force: bool = False) -> None:
        self.router.set_candidates(self.catalog.text_models(FALLBACK_MODELS))
        self.embed_router.set_candidates(self.catalog.embedding_models(EMBEDDING_FALLBACKS))

    def generate(self, contents, config=None):
        """generate_content with automatic model failover. Raises AllModelsFailed."""
        self.refresh_models()
        def _call(model):
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:
                if ("404" in str(exc) or "NOT_FOUND" in str(exc)) and model == self.router.preferred:
                    self._persist_switch = True
                raise
        return self.router.call(_call)

    # --- model discovery ---

    def list_models(self) -> list[str]:
        names: list[str] = []
        try:
            for m in self.client.models.list():
                name = getattr(m, "name", "") or ""
                name = name.replace("models/", "")
                if "flash" in name or "pro" in name:
                    # Prefer generateContent-capable models
                    methods = getattr(m, "supported_actions", None) or getattr(
                        m, "supported_generation_methods", None
                    )
                    if methods and "generateContent" not in str(methods):
                        continue
                    names.append(name)
        except Exception as exc:
            logger.warning("list_models failed: %s", exc)
        if not names:
            names = list(FALLBACK_MODELS)
        return names

    def pick_model(self, preferred: str | None = None) -> str:
        ranked = self.catalog.text_models(FALLBACK_MODELS)
        if preferred and preferred in ranked:
            self.model = preferred
        elif ranked:
            self.model = ranked[0]
        else:
            self.model = preferred or DEFAULT_MODEL
        self.router.preferred = self.model
        self.router.set_candidates(ranked)
        self.router.current = self.model
        return self.model

    def embed(self, text: str) -> list[float] | None:
        try:
            result = self.embed_router.call(
                lambda m: self.client.models.embed_content(model=m, contents=text)
            )
            emb = getattr(result, "embeddings", None) or getattr(result, "embedding", None)
            if emb is None:
                return None
            # google-genai returns list[ContentEmbedding] or a single embedding
            if isinstance(emb, list):
                values = getattr(emb[0], "values", None) if emb else None
            else:
                values = getattr(emb, "values", None)
            if values is None and isinstance(emb, list) and emb and isinstance(emb[0], (list, tuple)):
                values = emb[0]
            return list(values) if values else None
        except Exception as exc:
            logger.debug("embed failed: %s", exc)
            return None

    # --- chat ---

    def _system_instruction(self, user_text: str) -> str:
        from pc_agent.config import load_config
        cfg = load_config()
        base = build_system_prompt(
            cfg.get("agent_name") or "Atlas",
            cfg.get("personality") or "Witty, warm, and sharp.",
        )
        if self.memory_context_fn:
            ctx = self.memory_context_fn(user_text) or ""
            if ctx.strip():
                base = base + "\n\n" + ctx
        # Optional live screen summary (privacy toggle)
        screen_summary = getattr(self, "screen_summary", None)
        if callable(screen_summary):
            try:
                s = screen_summary() or ""
            except Exception:
                s = ""
            if s.strip():
                base += "\n\n## Latest screen (awareness on)\n" + s.strip()
        return base

    def _tools_config(self):
        types = self._types
        if not self.tool_declarations:
            return None
        decls = []
        for d in self.tool_declarations:
            decls.append(
                types.FunctionDeclaration(
                    name=d["name"],
                    description=d.get("description", ""),
                    parameters=d.get("parameters"),
                )
            )
        return [types.Tool(function_declarations=decls)]

    def chat(self, user_text: str, image_bytes: bytes | None = None) -> str:
        """Send a user message, run tool loops, return final assistant text."""
        types = self._types
        parts: list[Any] = [types.Part.from_text(text=user_text)]
        if image_bytes:
            parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            )
        user_msg = types.Content(role="user", parts=parts)
        self._history.append(user_msg)

        tools = self._tools_config()
        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction(user_text),
            tools=tools,
            temperature=0.4,
        )

        max_rounds = 12
        for _ in range(max_rounds):
            try:
                response = self.generate(self._history, config)
            except AllModelsFailed:
                if self._history and self._history[-1] is user_msg:
                    self._history.pop()
                return ALL_FAILED_MSG
            except Exception:
                logger.exception("generate_content failed")
                return "Something went wrong talking to the model. Try that again in a moment."

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                return "No response from the model."

            content = candidate.content
            self._history.append(content)

            fn_calls = [
                p.function_call
                for p in (content.parts or [])
                if getattr(p, "function_call", None)
            ]
            if not fn_calls:
                # Final text answer
                texts = [
                    p.text
                    for p in (content.parts or [])
                    if getattr(p, "text", None)
                ]
                return "\n".join(texts).strip() or "(no text)"

            # Execute tools and feed results back
            result_parts = []
            for fc in fn_calls:
                name = fc.name
                args = dict(fc.args or {})
                handler = self.tool_handlers.get(name)
                if handler is None:
                    result = {"ok": False, "error": f"unknown tool: {name}"}
                else:
                    try:
                        result = handler(args)
                    except Exception as exc:
                        logger.exception("tool %s failed", name)
                        result = {"ok": False, "error": str(exc)}
                result_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response=result if isinstance(result, dict) else {"result": result},
                    )
                )
            self._history.append(types.Content(role="user", parts=result_parts))

        return "Stopped after too many tool rounds. Try a simpler request."

    def summarize(self, messages: list[dict]) -> str:
        """Compress old conversation history into a short summary."""
        if not messages:
            return ""
        transcript = "\n".join(
            f"{m['role']}: {m['content'][:500]}" for m in messages
        )
        prompt = (
            "Summarize this conversation for future context. Keep durable facts, "
            "preferences, decisions, and open tasks. Be concise (under 250 words).\n\n"
            + transcript
        )
        try:
            types = self._types
            response = self.generate(prompt, types.GenerateContentConfig(temperature=0.2))
            return (response.text or "").strip()
        except Exception as exc:
            logger.warning("summarize failed: %s", exc)
            return transcript[:800]

    def reset_session(self) -> None:
        self._history.clear()
