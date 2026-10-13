from pc_agent.core import safety


def test_halt_and_resume():
    safety.resume()
    assert not safety.is_halted()
    safety.halt("test")
    assert safety.is_halted()
    safety.resume()
    assert not safety.is_halted()


def test_confirm_denies_without_ui(monkeypatch):
    safety.resume()
    safety.set_confirm_callback(None)
    assert safety.needs_confirmation("delete_file")
    assert not safety.confirm("delete_file", "rm stuff")


def test_confirm_callback(monkeypatch):
    safety.resume()
    safety.set_confirm_callback(lambda c, d: True)
    assert safety.confirm("shutdown", "reboot now")
    safety.set_confirm_callback(lambda c, d: False)
    assert not safety.confirm("shutdown", "reboot now")
