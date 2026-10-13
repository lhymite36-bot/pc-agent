"""UI contains mic Talk button and Voice mode toggle (shipped HTML)."""

from pc_agent.ui.web_ui import INDEX_HTML, create_app
from fastapi.testclient import TestClient


def test_mic_button_and_voice_toggle_render():
    html = INDEX_HTML
    assert 'id="mic-btn"' in html
    assert "Talk" in html or "aria-label=\"Talk\"" in html
    assert 'id="voice-mode-toggle"' in html
    assert "Voice mode" in html
    assert "/ws/voice" in html
    assert "getUserMedia" in html
    assert 'id="view-voice"' in html
    assert "testSpeak" in html
    assert "voice-enabled" in html


def test_voice_api_routes_exist():
    app = create_app(
        chat_fn=lambda t: "ok",
        memory=__import__("pc_agent.core.memory", fromlist=["MemoryStore"]).MemoryStore(
            path=__import__("tempfile").mktemp(suffix=".db")
        ),
        voice=None,
        get_settings=lambda: {},
        update_settings=lambda b: None,
        set_key=lambda k: None,
        halt_fn=lambda: None,
    )
    # Routes registered
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/ws/voice" in paths
    assert "/api/voice/test" in paths
    assert "/api/voice/mode" in paths
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="mic-btn"' in r.text
    assert 'id="voice-mode-toggle"' in r.text
