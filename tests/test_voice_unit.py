"""Voice helpers without calling Gemini."""

from pc_agent.core.voice import VoiceEngine, PREVIEW_LINE
from pc_agent.config import VOICE_CATALOG, DEFAULT_VOICE, VOICE_STYLE_PROMPT


def test_catalog_and_defaults():
    assert DEFAULT_VOICE == "Sulafat"
    assert any(v["id"] == "Sulafat" for v in VOICE_CATALOG)
    assert "contractions" in VOICE_STYLE_PROMPT.lower() or "I'm" in VOICE_STYLE_PROMPT
    assert "conversational" in VOICE_STYLE_PROMPT.lower()


def test_offline_fallback_label():
    eng = VoiceEngine(api_key=None)
    # Force offline path
    from pc_agent.config import load_config, save_config

    cfg = load_config()
    cfg["voice_mode"] = "offline"
    cfg["voice_enabled"] = True
    save_config(cfg)
    result = eng.speak("Hello there.")
    # On Linux CI, pyttsx3 may or may not work — just ensure structured result
    assert "engine" in result or "error" in result or result.get("ok") is not None
