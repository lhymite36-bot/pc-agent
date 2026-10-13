from datetime import datetime, timezone, timedelta
from pc_agent.core import quota
from pc_agent.core.local_stt import available, model_dir


def test_parse_retry_hours():
    assert abs(quota.parse_retry_seconds("Please retry in 7h26m14s.") - (7*3600+26*60+14)) < 1
    assert abs(quota.parse_retry_seconds("retry in 37s") - 37) < 0.1
    assert abs(quota.parse_retry_seconds("retryDelay: '40s'") - 40) < 0.1


def test_quota_notice_local_time():
    msg = quota.format_quota_notice("429 RESOURCE_EXHAUSTED free_tier retry in 3600s")
    assert "Free Gemini quota is used up until about" in msg
    assert "billing" not in msg.lower()
    assert "local" in msg
    msg2 = quota.format_quota_notice("429", model_resets={"gemini-3.6-flash": 3600})
    assert "gemini-3.6-flash" in msg2


def test_is_quota_exhausted():
    assert quota.is_quota_exhausted("429 RESOURCE_EXHAUSTED free tier 20 requests/day")
    assert not quota.is_quota_exhausted("404 NOT_FOUND")


def test_local_stt_module_paths():
    # Don't download the model in CI — just ensure helpers work
    d = model_dir()
    assert d.exists()
    ok, detail = available()
    assert isinstance(ok, bool) and isinstance(detail, str)


def test_chat_failover_429_cools_with_delay(tmp_path):
    from pc_agent.core import models as M

    t = [0.0]
    h = M.ModelHealth(tmp_path / "h.json", clock=lambda: t[0])
    r = M.ModelRouter("text", ["a", "b"], preferred="a", sleep=lambda s: None, clock=lambda: t[0], health=h)
    calls = []

    def fn(m):
        calls.append(m)
        if m == "a":
            raise Exception("429 RESOURCE_EXHAUSTED Please retry in 7h26m.")
        return "ok"

    assert r.call(fn) == "ok"
    assert calls == ["a", "b"]
    assert not h.usable("a")
    # still cooling after 1h
    t[0] += 3600
    assert not h.usable("a")
    t[0] += 7 * 3600
    assert h.usable("a")
