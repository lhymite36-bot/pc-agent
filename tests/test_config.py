from pc_agent import config


def test_api_key_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    config.clear_api_key()
    assert config.get_api_key() in (None, "")
    config.set_api_key("test-key-123")
    assert config.get_api_key() == "test-key-123"
    config.clear_api_key()
