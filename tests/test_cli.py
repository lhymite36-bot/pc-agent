"""Terminal mode / CLI helpers (Linux-friendly)."""

from pc_agent.cli import selftest, TerminalApp
from pc_agent.core.memory import MemoryStore


def test_selftest_runs():
    # On Linux CI, sounddevice may or may not work — selftest should still exit 0/1 cleanly
    code = selftest()
    assert code in (0, 1)


def test_memory_wal_mode(tmp_path):
    store = MemoryStore(path=tmp_path / "w.db")
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    store.remember("wal-ok", kind="note")
    # Second connection can read (shared DB)
    store2 = MemoryStore(path=tmp_path / "w.db")
    hits = store2.recall("wal-ok", use_embeddings=False)
    assert any("wal-ok" in h.content for h in hits)
    store.close()
    store2.close()


def test_terminal_commands_no_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    class FakeBrain:
        model_note = "Using test"
        def list_models(self):
            return ["gemini-3.6-flash"]
        def pick_model(self, n):
            return n

    class FakeAgent:
        brain = FakeBrain()
        memory = MemoryStore(path=tmp_path / "m.db")
        voice = None

        def chat_stream(self, text):
            yield "Hello "
            yield "from CLI"

    app = TerminalApp(FakeAgent())
    assert app.handle_command("/help") is True
    assert app.handle_command("/model") is True
    assert app.handle_command("/memory") is True
    assert app.handle_command("/text") is True
    # unknown
    assert app.handle_command("/nope") is True
    # not a command
    assert app.handle_command("hello") is False


def test_cli_starts_and_quits_without_key(tmp_path):
    import os, subprocess, sys
    env = dict(os.environ, XDG_DATA_HOME=str(tmp_path))
    r = subprocess.run([sys.executable, "-m", "pc_agent.cli_main"], input="/quit\n",
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    assert "Bye." in r.stdout
