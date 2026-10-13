from pc_agent.tools import files, shell
from pc_agent.core import safety


def test_write_read_list_delete(tmp_path):
    safety.resume()
    safety.set_confirm_callback(lambda c, d: True)
    p = tmp_path / "a.txt"
    assert files.write_file({"path": str(p), "content": "hello"})["ok"]
    assert files.read_file({"path": str(p)})["content"] == "hello"
    listing = files.list_files({"path": str(tmp_path)})
    assert listing["ok"] and any(e["name"] == "a.txt" for e in listing["entries"])
    assert files.delete_file({"path": str(p)})["ok"]


def test_shell_echo():
    safety.resume()
    r = shell.run_shell({"command": "echo hello", "shell": "bash", "timeout_sec": 10})
    assert r["ok"]
    assert "hello" in r["stdout"]


def test_shell_dangerous_denied():
    safety.resume()
    safety.set_confirm_callback(lambda c, d: False)
    r = shell.run_shell({"command": "rm -rf /tmp/nope", "shell": "bash"})
    assert not r["ok"]
