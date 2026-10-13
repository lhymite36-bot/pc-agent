from pc_agent.core import skills
from pc_agent.core import plugins
from pc_agent.config import skills_dir, plugins_dir


def test_save_list_run_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    # reload paths via skills_dir which uses data_dir
    data = skills.save_skill(
        "Open Notepad",
        "demo",
        [{"action": "open_app", "args": {"target": "notepad.exe"}, "expect": "notepad open"}],
    )
    assert data["id"]
    listed = skills.list_skills()
    assert any(s["id"] == data["id"] for s in listed)
    got = skills.get_skill(data["id"])
    assert got and got["name"] == "Open Notepad"
    assert skills.delete_skill(data["id"])


def test_plugin_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    root = plugins_dir()
    (root / "echo_tool.py").write_text(
        '''
DECLARATIONS = [{
  "name": "echo_tool",
  "description": "Echo",
  "parameters": {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
}]
def _echo(args):
    return {"ok": True, "text": args.get("text")}
HANDLERS = {"echo_tool": _echo}
''',
        encoding="utf-8",
    )
    decls, hands = plugins.load_plugins()
    assert any(d["name"] == "echo_tool" for d in decls)
    assert hands["echo_tool"]({"text": "hi"})["text"] == "hi"
