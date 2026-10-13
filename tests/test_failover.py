from pc_agent.config import DEFAULT_MODEL, load_config, config_path
from pc_agent.core import models as M


class Err(Exception):
    pass


def test_default_model():
    assert DEFAULT_MODEL == "gemini-3.8-flash"


def test_rank_text_models():
    names = ["models/gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.8-flash-tts",
             "gemini-3.8-pro", "gemini-3.8-live", "text-embedding-004", "gemini-2.5-flash-image", "veo-3",
             "gemini-2.5-pro"]
    r = M.rank_text_models(names)
    assert r[0] == "gemini-3.8-flash"
    assert r[1] == "gemini-3.8-pro"
    assert "gemini-2.5-flash" in r
    for bad in ("lite", "tts", "live", "image", "embedding", "veo"):
        assert not any(bad in x for x in r)


def test_rank_embeddings():
    r = M.rank_embedding_models(["text-embedding-004", "gemini-embedding-001", "gemini-3.8-flash"])
    assert r[0] == "gemini-embedding-001" and "gemini-3.8-flash" not in r


def test_retryable():
    assert M.is_retryable(Err("503 UNAVAILABLE high demand"))
    assert M.is_retryable(Err("429 RESOURCE_EXHAUSTED"))
    assert M.is_retryable(Err("404 NOT_FOUND"))
    assert M.is_retryable(Err("500 INTERNAL"))
    assert not M.is_retryable(Err("400 INVALID_ARGUMENT"))


def _router(clock):
    return M.ModelRouter("text", ["a", "b", "c"], preferred="a", sleep=lambda s: None, clock=clock)


def test_failover_retry_then_next_and_cooldown():
    t = [0.0]
    r = _router(lambda: t[0])
    calls = []
    down = {"a"}

    def fn(m):
        calls.append(m)
        if m in down:
            raise Err("503 UNAVAILABLE")
        return m

    assert r.call(fn) == "b"
    assert calls == ["a", "a", "b"]  # one retry, then next
    calls.clear()
    assert r.call(fn) == "b" and calls == ["b"]  # sticky during cooldown
    t[0] += M.COOLDOWN_SEC + 1
    down.clear()
    calls.clear()
    assert r.call(fn) == "a" and calls == ["a"]  # preferred retried after cooldown
    assert "a" in r.last_note


def test_all_fail_friendly():
    r = _router(lambda: 0.0)
    try:
        r.call(lambda m: (_ for _ in ()).throw(Err("503 UNAVAILABLE")))
        assert False
    except M.AllModelsFailed:
        pass
    assert r.last_note == M.ALL_FAILED_MSG
    assert "503" not in M.ALL_FAILED_MSG


def test_non_retryable_raises():
    r = _router(lambda: 0.0)
    try:
        r.call(lambda m: (_ for _ in ()).throw(Err("400 bad")))
        assert False
    except Err:
        pass


def test_catalog_cache():
    class Mdl:
        def __init__(self, n): self.name = n; self.supported_actions = ["generateContent"]
    class Client:
        n = 0
        class models:
            pass
    calls = {"n": 0}
    class Models:
        def list(self):
            calls["n"] += 1
            return [Mdl("models/gemini-3.8-flash"), Mdl("models/gemini-3.8-pro")]
    c = Client(); c.models = Models()
    t = [0.0]
    cat = M.ModelCatalog(c, clock=lambda: t[0])
    assert cat.text_models([])[0] == "gemini-3.8-flash"
    cat.text_models([])
    assert calls["n"] == 1
    t[0] += M.LIST_TTL_SEC + 1
    cat.text_models([])
    assert calls["n"] == 2


def test_load_config_bom(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    config_path().write_bytes(b'\xef\xbb\xbf{"agent_name": "Bom"}')
    assert load_config()["agent_name"] == "Bom"


def test_brain_failover_persists_on_404(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from pc_agent.core import brain as B

    class Resp:
        text = "hi"
    class Models:
        def generate_content(self, model, contents, config=None):
            if model == "gemini-3.8-flash":
                raise Err("404 NOT_FOUND no longer available")
            return Resp()
        def list(self):
            return []
    br = B.Brain.__new__(B.Brain)
    class C: pass
    br.client = C(); br.client.models = Models()
    br.model = "gemini-3.8-flash"
    br.catalog = M.ModelCatalog(br.client)
    br.router = M.ModelRouter("text", ["gemini-3.8-flash", "gemini-3.8-pro"], preferred="gemini-3.8-flash",
                              on_switch=br._on_switch, sleep=lambda s: None)
    br.embed_router = M.ModelRouter("embed", ["e"], sleep=lambda s: None)
    monkeypatch.setattr(B, "FALLBACK_MODELS", ["gemini-3.8-flash", "gemini-3.8-pro"])
    assert br.generate("x").text == "hi"
    assert br.model == "gemini-3.8-pro"
    assert load_config()["model"] == "gemini-3.8-pro"
    assert "gemini-3.8-pro" in br.model_note
