"""Memory store unit tests (Linux-friendly, no GUI)."""

from __future__ import annotations

import math

import pytest

from pc_agent.core.memory import MemoryStore


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(path=tmp_path / "t.db")
    yield s
    s.close()


def test_remember_recall_forget(store: MemoryStore):
    a = store.remember("User's name is Pedro", kind="fact", tags="name")
    store.remember("Prefers dark mode", kind="preference", tags="ui")
    store.remember("Working on Receipts app", kind="note", tags="project")

    hits = store.recall("Pedro name", use_embeddings=False)
    assert any("Pedro" in h.content for h in hits)

    prefs = store.recall("dark mode", kinds=["preference"], use_embeddings=False)
    assert prefs and prefs[0].kind == "preference"

    n = store.forget(memory_id=a.id)
    assert n == 1
    assert store.get_memory(a.id) is None


def test_unique_upsert(store: MemoryStore):
    store.remember("Call me Pedro", kind="preference", tags="a")
    store.remember("Call me Pedro", kind="preference", tags="b")
    items = store.list_memories(kinds=["preference"])
    assert len([i for i in items if i.content == "Call me Pedro"]) == 1
    assert items[0].tags == "b"


def test_auto_extract(store: MemoryStore):
    saved = store.auto_extract("Remember that my name is Pedro and I prefer short replies.")
    assert saved
    kinds = {s.kind for s in saved}
    assert "fact" in kinds or "preference" in kinds


def test_conversation_and_summary(store: MemoryStore):
    for i in range(5):
        store.add_message("user", f"msg {i}")
        store.add_message("assistant", f"ok {i}")
    assert store.message_count() == 10
    recent = store.get_recent_messages(4)
    assert len(recent) == 4
    store.add_summary("User chatted casually.", recent[-1]["id"])
    assert "casually" in (store.get_latest_summary() or "")


def test_tasks_and_episodes(store: MemoryStore):
    tid = store.enqueue_task("remind stretch", payload={"prompt": "remind"}, run_at="2000-01-01T00:00:00+00:00")
    due = store.due_tasks("2020-01-01T00:00:00+00:00")
    assert any(t["id"] == tid for t in due)
    store.update_task(tid, status="done")
    eid = store.log_episode("Installed app", detail="ok", outcome="success")
    assert eid > 0
    assert store.list_episodes()


def test_embedding_recall(store: MemoryStore):
    def fake_embed(text: str):
        # Simple bag-of-words hash embedding
        vec = [0.0] * 8
        for i, ch in enumerate(text.lower()):
            vec[i % 8] += (ord(ch) % 13) / 13.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    store.set_embedder(fake_embed)
    store.remember("The sky is blue today", kind="note")
    store.remember("Database migration finished", kind="note")
    hits = store.recall("sky blue color", use_embeddings=True)
    assert hits
    assert "sky" in hits[0].content.lower() or hits[0].score >= 0


def test_memory_tools_dispatch(store: MemoryStore):
    r = store.dispatch_tool("remember", {"content": "Likes espresso", "kind": "preference"})
    assert r["ok"]
    r2 = store.dispatch_tool("recall", {"query": "espresso"})
    assert r2["ok"] and r2["results"]
    mid = r2["results"][0]["id"]
    r3 = store.dispatch_tool("forget", {"memory_id": mid})
    assert r3["ok"] and r3["deleted"] == 1


def test_build_memory_context(store: MemoryStore):
    store.remember("User's name is Pedro", kind="fact")
    ctx = store.build_memory_context("What is my name?")
    assert "Pedro" in ctx


def test_remember_is_fast_and_embeds_in_background(store):
    calls = []
    store.set_embedder(lambda t: calls.append(t) or [1.0, 0.0])
    store.remember("Likes tea", kind="preference")
    assert calls == []  # no network on the request path
    assert store.embed_pending() == 1 and calls == ["Likes tea"]
    ctx = store.build_memory_context("tea")
    assert "tea" in ctx and len(calls) == 1  # prompt recall is FTS-only
