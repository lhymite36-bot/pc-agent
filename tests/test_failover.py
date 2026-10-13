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
    assert r[1] == "gemini-2.5-flash"  # all flash before any pro
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



def test_rank_flash_before_pro_and_no_customtools():
    names = ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
             "gemini-2.5-pro", "gemini-pro-latest", "gemini-3.7-flash", "gemini-3.6-flash"]
    r = M.rank_text_models(names)
    assert r[:3] == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    assert not any("customtools" in x for x in r)
    assert all("flash" not in x for x in r[3:])


def test_404_and_429_not_retried_and_persisted(tmp_path):
    t = [0.0]
    h = M.ModelHealth(tmp_path / "h.json", clock=lambda: t[0])
    r = M.ModelRouter("text", ["a", "b", "c"], preferred="a", sleep=lambda s: None, clock=lambda: t[0], health=h)
    calls = []

    def fn(m):
        calls.append(m)
        if m == "a":
            raise Err("404 NOT_FOUND")
        if m == "b":
            raise Err("429 RESOURCE_EXHAUSTED retryDelay: '40s'")
        return m

    assert r.call(fn) == "c"
    assert calls == ["a", "b", "c"]  # no retries
    # Persisted across "restart"
    h2 = M.ModelHealth(tmp_path / "h.json", clock=lambda: t[0])
    assert not h2.usable("a") and not h2.usable("b") and h2.usable("c")
    t[0] += 41
    assert h2.usable("b") and not h2.usable("a")


def test_retry_delay_parse():
    assert M.retry_delay_sec(Err("retryDelay': '12s'")) == 12.0
    assert M.retry_delay_sec(Err("Please retry in 7.5s")) == 7.5
    assert M.retry_delay_sec(Err("429")) == M.DEFAULT_429_COOLDOWN_SEC


def test_503_single_quick_retry():
    sleeps = []
    r = M.ModelRouter("text", ["a", "b"], preferred="a", sleep=sleeps.append, clock=lambda: 0.0)
    calls = []

    def fn(m):
        calls.append(m)
        if m == "a":
            raise Err("503 UNAVAILABLE")
        return m

    assert r.call(fn) == "b"
    assert calls == ["a", "a", "b"] and sleeps == [M.RETRY_503_DELAY_SEC]
    assert M.RETRY_503_DELAY_SEC <= 0.3


def test_catalog_never_fetches_in_request_path(tmp_path):
    calls = {"n": 0}

    class Mdl:
        def __init__(self, n): self.name = n; self.supported_actions = ["generateContent"]

    class Models:
        def list(self):
            calls["n"] += 1
            return [Mdl("models/gemini-3.8-flash"), Mdl("models/gemini-3.8-pro")]

    class C: pass
    c = C(); c.models = Models()
    cat = M.ModelCatalog(c, cache_path=tmp_path / "m.json")
    assert cat.text_models(["fb"]) == ["fb"] and calls["n"] == 0
    cat.refresh()
    assert calls["n"] == 1 and cat.text_models([])[0] == "gemini-3.8-flash"
    cat2 = M.ModelCatalog(c, cache_path=tmp_path / "m.json")  # cache survives restart
    assert cat2.text_models([])[0] == "gemini-3.8-flash" and calls["n"] == 1


def test_load_config_bom(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    config_path().write_bytes(b'\xef\xbb\xbf{"agent_name": "Bom"}')
    assert load_config()["agent_name"] == "Bom"


class _Part:
    def __init__(self, text=None, fc=None):
        self.text = text; self.function_call = fc; self.thought = False


class _Chunk:
    def __init__(self, text):
        class Cand: pass
        class Content: pass
        c = Cand(); c.content = Content(); c.content.parts = [_Part(text)]
        self.candidates = [c]; self.text = text


def _fake_brain(tmp_path, monkeypatch, behaviour):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from pc_agent.core import brain as B
    from google.genai import types

    class Models:
        def generate_content(self, model, contents, config=None):
            behaviour(model)
            return _Chunk("hi")
        def generate_content_stream(self, model, contents, config=None):
            behaviour(model)
            yield _Chunk("Hello ")
            yield _Chunk("there.")
        def list(self):
            return []

    br = B.Brain.__new__(B.Brain)
    class C: pass
    br.client = C(); br.client.models = Models()
    br._types = types
    br.model = "gemini-3.8-flash"
    br.tool_handlers = {}; br.tool_declarations = []; br.memory_context_fn = None
    br._history = []; br._tools_cache = None; br._persist_switch = False
    br.health = M.ModelHealth(tmp_path / "h.json")
    br.router = M.ModelRouter("text", ["gemini-3.8-flash", "gemini-3.6-flash"], preferred="gemini-3.8-flash",
                              on_switch=br._on_switch, sleep=lambda s: None, health=br.health)
    br.embed_router = M.ModelRouter("embed", ["e"], sleep=lambda s: None, health=br.health)
    return br


def test_brain_404_persists_model(tmp_path, monkeypatch):
    def beh(model):
        if model == "gemini-3.8-flash":
            raise Err("404 NOT_FOUND no longer available")
    br = _fake_brain(tmp_path, monkeypatch, beh)
    assert br.generate("x").text == "hi"
    assert br.model == "gemini-3.6-flash"
    assert load_config()["model"] == "gemini-3.6-flash"
    assert "gemini-3.6-flash" in br.model_note


def test_stream_fails_over_before_first_chunk(tmp_path, monkeypatch):
    def beh(model):
        if model == "gemini-3.8-flash":
            raise Err("503 UNAVAILABLE high demand")
    br = _fake_brain(tmp_path, monkeypatch, beh)
    out = list(br.chat_stream("hey"))
    assert "".join(out) == "Hello there."
    assert br.router.current == "gemini-3.6-flash"


def test_stream_all_fail_friendly(tmp_path, monkeypatch):
    def beh(model):
        raise Err("503 UNAVAILABLE")
    br = _fake_brain(tmp_path, monkeypatch, beh)
    out = "".join(br.chat_stream("hey"))
    assert out == M.ALL_FAILED_MSG and br._history == []


def test_thinking_config_levels():
    from google.genai import types
    from pc_agent.core.brain import thinking_config
    assert thinking_config(types, "gemini-3.8-flash", "minimal").thinking_level is not None
    assert thinking_config(types, "gemini-2.5-flash", "minimal").thinking_budget == 0
    assert thinking_config(types, "gemini-2.5-flash", "high").thinking_budget > 1000


def test_trim_history_keeps_pairs():
    from pc_agent.core.brain import _trim_history

    class P:
        def __init__(self, text=None, fr=None): self.text = text; self.function_response = fr
    class Cn:
        def __init__(self, role, parts): self.role = role; self.parts = parts
    h = []
    for i in range(20):
        h.append(Cn("user", [P(text=f"q{i}")]))
        h.append(Cn("model", [P(text=None)]))
        h.append(Cn("user", [P(fr={"ok": True})]))
    t = _trim_history(h, limit=10)
    assert len(t) <= 10 and t[0].parts[0].text and t[0].role == "user"


def test_sentence_splitter():
    from pc_agent.app import SentenceSplitter
    s = SentenceSplitter(); out = []
    for d in ["Sure thing. I", "'ll open it now! Then **save**", ".\nDone"]:
        out += s.feed(d)
    assert out[0].startswith("Sure thing.") and "Then save." in out
    assert s.flush() == "Done"
