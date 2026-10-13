
from pc_agent.core.offline_router import try_handle

def test_time():
    r = try_handle("what time is it")
    assert r and "It" in r

def test_open_app_patterns():
    # Don't actually launch — mock by checking None vs string on unknown
    assert try_handle("open chrome") is not None
    assert try_handle("open downloads") is not None
    assert try_handle("volume mute") is not None
    assert try_handle("take a screenshot") is not None
    assert try_handle("lock the pc") is not None
    assert try_handle("explain quantum physics in detail please") is None

def test_url():
    r = try_handle("open https://example.com")
    assert r and "Opening" in r
