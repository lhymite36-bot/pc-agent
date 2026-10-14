"""Exactly one voice engine per utterance; Gemini first, local only as fallback."""
import threading

import pytest

from pc_agent.core import voice as voice_mod


@pytest.fixture
def cfg(monkeypatch):
    store = {"voice_enabled": True, "voice_mode": "live"}
    monkeypatch.setattr(voice_mod, "load_config", lambda: dict(store))
    monkeypatch.setattr(voice_mod, "save_config", lambda c: store.update(c))
    return store


def _engine(calls, live_ok=True, tts_ok=True, tts_err="503"):
    eng = voice_mod.VoiceEngine.__new__(voice_mod.VoiceEngine)
    eng._stop = threading.Event()
    eng._speak_lock = threading.Lock()
    eng.last_engine = None
    eng.engine_log = []
    eng._client = object()
    eng._with_style = lambda t: t

    def live(t, on_chunk=None):
        calls.append("live")
        return {"ok": live_ok, "engine": "live", "error": "" if live_ok else "503"}

    def tts(t, on_chunk=None):
        calls.append("tts")
        return {"ok": tts_ok, "engine": "tts", "error": "" if tts_ok else tts_err}

    def off(t):
        calls.append("offline")
        return {"ok": True, "engine": "offline"}

    eng._speak_live, eng._speak_tts, eng._speak_offline = live, tts, off
    return eng


def test_gemini_live_only(cfg):
    calls = []
    r = _engine(calls).speak("hello there")
    assert calls == ["live"] and r["engine"] == "live"


def test_gemini_tts_when_live_fails_no_local(cfg):
    calls = []
    r = _engine(calls, live_ok=False).speak("hi")
    assert r["engine"] == "tts" and "offline" not in calls


def test_local_only_when_gemini_fails(cfg):
    calls = []
    r = _engine(calls, live_ok=False, tts_ok=False,
                tts_err="429 RESOURCE_EXHAUSTED retry in 60s").speak("hi")
    assert calls.count("offline") == 1 and r["engine"] == "offline"
    assert cfg["voice_mode"] == "live"  # never permanently flipped to robotic
    assert cfg.get("voice_gemini_paused_until")


def test_one_engine_per_utterance_concurrent(cfg):
    calls = []
    eng = _engine(calls)
    ts = [threading.Thread(target=eng.speak, args=(f"s{i}",)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(eng.engine_log) == 8 and set(eng.engine_log) == {"live"}
    assert "offline" not in calls


def test_legacy_sticky_offline_is_ignored(cfg):
    cfg["voice_mode"] = "offline"
    assert voice_mod.VoiceEngine.mode() == "live"
    cfg["voice_offline_user"] = True
    assert voice_mod.VoiceEngine.mode() == "offline"


def test_cli_chat_does_not_add_second_voice(monkeypatch):
    from pc_agent import cli

    said, streamed = [], []

    class FakeAgent:
        def chat_stream(self, text):
            streamed.append(text)  # agent's own worker speaks this (Gemini)
            yield "Sure thing."

        def say(self, text):
            said.append(text)

    app = cli.TerminalApp(FakeAgent())
    monkeypatch.setattr("pc_agent.core.offline_router.try_handle", lambda t: None)
    app.chat_line("tell me a joke")
    assert streamed == ["tell me a joke"] and said == []


def test_web_chat_fn_is_silent():
    import inspect
    from pc_agent import app as app_mod
    src = inspect.getsource(app_mod.Agent.chat)
    assert "speak=False" in src
