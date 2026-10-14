"""Free multi-provider failover (no billing), Ollama probe, reply cache."""
import json
import time

import httpx
import pytest

from pc_agent.core import providers, reply_cache


@pytest.fixture(autouse=True)
def tmpdata(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(reply_cache, "data_dir", lambda: tmp_path)
    return tmp_path


def _resp(code, payload=None, headers=None):
    return httpx.Response(code, json=payload or {}, headers=headers or {},
                          request=httpx.Request("POST", "http://x"))


def _ok(text):
    return _resp(200, {"choices": [{"message": {"content": text}}]})


def test_groq_429_fails_over_to_openrouter_and_cools():
    providers.add_key("groq", "g1")
    providers.add_key("openrouter", "o1")
    seen = []

    def post(url, **kw):
        seen.append(url)
        if "groq" in url:
            return _resp(429, {"error": "rate"}, {"retry-after": "120"})
        return _ok("hi from openrouter")

    t = time.time()
    assert providers.chat([{"role": "user", "content": "hi"}], post=post) == "hi from openrouter"
    assert time.time() - t < 1
    assert providers.cooling_until("groq", 0) > time.time() + 100
    # next call skips groq entirely
    seen.clear()
    providers.chat([{"role": "user", "content": "hi"}], post=post)
    assert all("groq" not in u for u in seen)


def test_timeout_fails_over():
    providers.add_key("groq", "g1")
    providers.add_key("mistral", "m1")

    def post(url, **kw):
        assert kw["timeout"].read <= 10
        if "groq" in url:
            raise httpx.ReadTimeout("slow")
        return _ok("mistral here")

    assert providers.chat([{"role": "user", "content": "x"}], post=post) == "mistral here"


def test_multiple_keys_per_provider_rotate():
    providers.add_key("groq", "k1")
    providers.add_key("groq", "k2")
    auth = []

    def post(url, **kw):
        auth.append(kw["headers"]["Authorization"])
        if kw["headers"]["Authorization"].endswith("k1"):
            return _resp(429, headers={"retry-after": "60"})
        return _ok("k2 works")

    assert providers.chat([{"role": "user", "content": "x"}], post=post) == "k2 works"
    assert auth[:2] == ["Bearer k1", "Bearer k2"]


def test_tool_calling_preserved():
    providers.add_key("groq", "g")
    calls = {"n": 0}

    def post(url, **kw):
        calls["n"] += 1
        body = kw["json"]
        assert body["tools"][0]["function"]["name"] == "open_app"
        if calls["n"] == 1:
            return _resp(200, {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "open_app", "arguments": json.dumps({"name": "notepad"})}}]}}]})
        assert body["messages"][-1]["role"] == "tool"
        return _ok("Opened Notepad.")

    ran = []
    tools = providers.to_openai_tools([
        {"name": "open_app", "description": "Open", "parameters": {"type": "OBJECT", "properties": {"name": {"type": "STRING"}}}},
        {"name": "speak", "description": "no"},
    ])
    assert len(tools) == 1 and tools[0]["function"]["parameters"]["type"] == "object"
    out = providers.chat([{"role": "user", "content": "open notepad"}], tools,
                         {"open_app": lambda a: ran.append(a) or {"ok": True}}, post=post)
    assert out == "Opened Notepad." and ran == [{"name": "notepad"}]


def test_exhausted_notice_free_only():
    providers.add_key("groq", "g")
    providers.mark_cooling("groq", 0, 3600)
    with pytest.raises(providers.ProvidersExhausted):
        providers.chat([{"role": "user", "content": "x"}], post=lambda *a, **k: _ok("no"))
    msg = providers.exhausted_notice(time.time() + 600)
    low = msg.lower()
    for bad in ("billing", "pay", "card", "upgrade", "credit"):
        assert bad not in low
    assert "/keys add openrouter" in msg and "Resets" in msg


def test_setup_lists_free_signups_no_card_only():
    s = providers.setup_text()
    for url in ("console.groq.com/keys", "openrouter.ai/keys", "console.mistral.ai", "dash.cloudflare.com", "aistudio.google.com"):
        assert url in s
    assert "cerebras" not in s.lower() and "github" not in s.lower()


def test_ollama_probe_is_fast_when_absent():
    from pc_agent.core import ollama
    ollama._probe_cache.clear()
    t = time.time()
    ok, models = ollama.available(host="http://10.255.255.1:11434")
    assert not ok and time.time() - t < 1.5
    t = time.time()
    ollama.available(host="http://10.255.255.1:11434")  # cached
    assert time.time() - t < 0.05


def test_reply_cache():
    assert reply_cache.put("What is photosynthesis?", "Plants turning light into sugar.")
    assert reply_cache.get("what is photosynthesis") == "Plants turning light into sugar."
    assert not reply_cache.put("what time is it?", "10:00")
    assert not reply_cache.put("open chrome", "Opened.")


def test_cli_keys_add_provider(monkeypatch):
    from pc_agent import cli

    class A:
        def say(self, t): pass
    app = cli.TerminalApp(A())
    out = []
    monkeypatch.setattr(app, "print", lambda s: out.append(s))
    app.handle_command("/keys add groq gsk_test123")
    assert providers.list_keys("groq") == ["gsk_test123"]
    app.handle_command("/setup")
    assert any("console.groq.com" in o for o in out)


def test_agent_free_fallback_chain(monkeypatch):
    from pc_agent import app as app_mod
    from pc_agent.core import ollama
    ag = app_mod.Agent.__new__(app_mod.Agent)
    ag.brain = None
    ag._declarations, ag._handlers = [], {}

    class M:
        def get_recent_messages(self, n): return []
        def build_memory_context(self, t): return ""
    ag.memory = M()
    monkeypatch.setattr(ollama, "available", lambda timeout=0.8: (False, []))
    # nothing configured -> notice (fast, no hang)
    t = time.time()
    out = "".join(ag._free_fallback("explain gravity", None))
    assert "free" in out.lower() and time.time() - t < 2
    # provider configured -> its answer
    providers.add_key("groq", "g")
    monkeypatch.setattr(providers, "_default_post", lambda url, **kw: _ok("Gravity pulls."))
    assert "".join(ag._free_fallback("explain gravity?", None)) == "Gravity pulls."
    # provider now out -> cached answer
    providers.mark_cooling("groq", 0, 3600)
    assert "".join(ag._free_fallback("explain gravity?", None)) == "Gravity pulls."
