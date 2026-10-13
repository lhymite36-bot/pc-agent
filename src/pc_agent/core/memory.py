"""Long-term memory, conversation history, task/episode log (SQLite).

First-class features:
- remember / recall / forget tools the model calls itself
- automatic extraction of durable facts & preferences from chat
- recall into each prompt via keyword FTS + optional Gemini embeddings
- conversation history across restarts with rolling summarization
- task / episode log
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from pc_agent.config import EMBEDDING_MODEL, db_path

logger = logging.getLogger("pc_agent.memory")

# Soft caps for prompt injection
MAX_RECALL_ITEMS = 12
MAX_HISTORY_MESSAGES = 40
SUMMARIZE_AFTER = 60  # messages before we compress older ones
SUMMARY_KEEP_RECENT = 20


@dataclass
class MemoryItem:
    id: int
    kind: str  # fact | preference | note | episode
    content: str
    tags: str
    score: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "tags": self.tags,
            "score": self.score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MemoryStore:
    """Thread-safe SQLite memory store."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embed_fn: Callable[[str], list[float] | None] | None = None
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        self._init_schema()

    def set_embedder(self, fn: Callable[[str], list[float] | None] | None) -> None:
        """Optional Gemini embedding function: text -> vector or None."""
        self._embed_fn = fn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- schema ---

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'note',
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(kind, content)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags, content='memories', content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    up_to_message_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    run_at TEXT,
                    cron TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            c.commit()

    # --- helpers ---

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _pack_embedding(vec: list[float] | None) -> bytes | None:
        if not vec:
            return None
        import struct

        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack_embedding(blob: bytes | None) -> list[float] | None:
        if not blob:
            return None
        import struct

        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _embed(self, text: str) -> list[float] | None:
        if self._embed_fn is None:
            return None
        try:
            return self._embed_fn(text)
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def _query_vec(self, query: str) -> list[float] | None:
        cache = self.__dict__.setdefault("_qcache", {})
        if query in cache:
            return cache[query]
        vec = self._embed(query)
        if vec:
            if len(cache) > 256:
                cache.clear()
            cache[query] = vec
        return vec

    def embed_pending(self, limit: int = 50) -> int:
        """Background: compute embeddings for memories that lack one."""
        if self._embed_fn is None:
            return 0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content FROM memories WHERE embedding IS NULL ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        n = 0
        for r in rows:
            vec = self._embed(r["content"])
            if not vec:
                break  # embedding unavailable; try later
            with self._lock:
                self._conn.execute("UPDATE memories SET embedding=? WHERE id=?", (self._pack_embedding(vec), r["id"]))
                self._conn.commit()
            n += 1
        return n

    # --- remember / recall / forget ---

    def remember(
        self,
        content: str,
        kind: str = "note",
        tags: str | Iterable[str] = "",
        embed_now: bool = False,
    ) -> MemoryItem:
        content = (content or "").strip()
        if not content:
            raise ValueError("content is empty")
        kind = (kind or "note").strip().lower()
        if kind not in {"fact", "preference", "note", "episode"}:
            kind = "note"
        if isinstance(tags, (list, tuple, set)):
            tags_s = ",".join(str(t).strip() for t in tags if str(t).strip())
        else:
            tags_s = str(tags or "").strip()
        now = self._now()
        emb = self._pack_embedding(self._embed(content)) if embed_now else None
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO memories(kind, content, tags, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, content) DO UPDATE SET
                    tags=excluded.tags,
                    embedding=COALESCE(excluded.embedding, memories.embedding),
                    updated_at=excluded.updated_at
                """,
                (kind, content, tags_s, emb, now, now),
            )
            self._conn.commit()
            row_id = cur.lastrowid
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id=?", (row_id,)
            ).fetchone()
            if row is None:
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE kind=? AND content=?",
                    (kind, content),
                ).fetchone()
        return self._row_to_item(row)

    def forget(self, memory_id: int | None = None, query: str | None = None) -> int:
        """Delete by id, or all keyword matches. Returns number deleted."""
        with self._lock:
            if memory_id is not None:
                cur = self._conn.execute(
                    "DELETE FROM memories WHERE id=?", (int(memory_id),)
                )
                self._conn.commit()
                return cur.rowcount
            if query:
                hits = self.recall(query, limit=50, use_embeddings=False)
                ids = [h.id for h in hits]
                if not ids:
                    return 0
                qmarks = ",".join("?" * len(ids))
                cur = self._conn.execute(
                    f"DELETE FROM memories WHERE id IN ({qmarks})", ids
                )
                self._conn.commit()
                return cur.rowcount
            return 0

    def recall(
        self,
        query: str,
        limit: int = MAX_RECALL_ITEMS,
        kinds: list[str] | None = None,
        use_embeddings: bool = True,
    ) -> list[MemoryItem]:
        query = (query or "").strip()
        if not query:
            return self.list_memories(limit=limit, kinds=kinds)

        scored: dict[int, MemoryItem] = {}

        # FTS5 keyword search
        fts_query = self._to_fts_query(query)
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT m.*, bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit * 3),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback LIKE search if FTS query is invalid
                like = f"%{query}%"
                rows = self._conn.execute(
                    """
                    SELECT *, 0.0 AS rank FROM memories
                    WHERE content LIKE ? OR tags LIKE ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (like, like, limit * 3),
                ).fetchall()

        for row in rows:
            item = self._row_to_item(row)
            # bm25 is lower=better; invert to a positive score
            rank = float(row["rank"]) if "rank" in row.keys() else 0.0
            item.score = 1.0 / (1.0 + abs(rank))
            if kinds and item.kind not in kinds:
                continue
            scored[item.id] = item

        # Embedding similarity (optional)
        if use_embeddings and self._embed_fn is not None:
            qvec = self._query_vec(query)
            if qvec:
                with self._lock:
                    all_rows = self._conn.execute(
                        "SELECT * FROM memories WHERE embedding IS NOT NULL"
                    ).fetchall()
                for row in all_rows:
                    if kinds and row["kind"] not in kinds:
                        continue
                    vec = self._unpack_embedding(row["embedding"])
                    if not vec:
                        continue
                    sim = self._cosine(qvec, vec)
                    if sim < 0.25:
                        continue
                    item = self._row_to_item(row, score=sim)
                    prev = scored.get(item.id)
                    if prev is None or item.score > prev.score:
                        scored[item.id] = item

        items = sorted(scored.values(), key=lambda m: m.score, reverse=True)
        return items[:limit]

    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0,
        kinds: list[str] | None = None,
        search: str | None = None,
    ) -> list[MemoryItem]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if kinds:
                qmarks = ",".join("?" * len(kinds))
                clauses.append(f"kind IN ({qmarks})")
                params.extend(kinds)
            if search:
                clauses.append("(content LIKE ? OR tags LIKE ?)")
                like = f"%{search}%"
                params.extend([like, like])
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.extend([limit, offset])
            rows = self._conn.execute(
                f"""
                SELECT * FROM memories {where}
                ORDER BY updated_at DESC LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    def count_memories(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def _row_to_item(self, row: sqlite3.Row, score: float = 0.0) -> MemoryItem:
        return MemoryItem(
            id=int(row["id"]),
            kind=row["kind"],
            content=row["content"],
            tags=row["tags"] or "",
            score=score,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _to_fts_query(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]{2,}", text)
        if not tokens:
            return '"' + text.replace('"', " ") + '"'
        return " OR ".join(tokens)

    # --- conversation history ---

    def add_message(self, role: str, content: str) -> int:
        now = self._now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)",
                (role, content, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_recent_messages(self, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content, created_at FROM messages
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    def get_latest_summary(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["summary"] if row else None

    def add_summary(self, summary: str, up_to_message_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries(summary, up_to_message_id, created_at) VALUES (?, ?, ?)",
                (summary, up_to_message_id, self._now()),
            )
            self._conn.commit()

    def message_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

    def needs_summarization(self) -> bool:
        return self.message_count() >= SUMMARIZE_AFTER

    def messages_to_summarize(self) -> list[dict]:
        """Older messages beyond SUMMARY_KEEP_RECENT."""
        with self._lock:
            total = self.message_count()
            if total <= SUMMARY_KEEP_RECENT:
                return []
            cut = total - SUMMARY_KEEP_RECENT
            rows = self._conn.execute(
                """
                SELECT id, role, content FROM messages
                ORDER BY id ASC LIMIT ?
                """,
                (cut,),
            ).fetchall()
        return [
            {"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows
        ]

    def drop_messages_up_to(self, message_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE id <= ?", (message_id,))
            self._conn.commit()

    # --- tasks / episodes ---

    def enqueue_task(
        self,
        title: str,
        payload: dict | None = None,
        run_at: str | None = None,
        cron: str | None = None,
    ) -> int:
        now = self._now()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO tasks(title, payload, status, run_at, cron, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
                """,
                (title, json.dumps(payload or {}), run_at, cron, now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY id DESC", (status,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tasks ORDER BY id DESC LIMIT 100"
                ).fetchall()
        return [dict(r) for r in rows]

    def update_task(self, task_id: int, **fields: Any) -> None:
        allowed = {"status", "last_error", "run_at", "payload", "title"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "payload" in updates and isinstance(updates["payload"], dict):
            updates["payload"] = json.dumps(updates["payload"])
        updates["updated_at"] = self._now()
        cols = ", ".join(f"{k}=?" for k in updates)
        with self._lock:
            self._conn.execute(
                f"UPDATE tasks SET {cols} WHERE id=?",
                (*updates.values(), task_id),
            )
            self._conn.commit()

    def due_tasks(self, now_iso: str | None = None) -> list[dict]:
        now_iso = now_iso or self._now()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM tasks
                WHERE status='pending' AND run_at IS NOT NULL AND run_at <= ?
                ORDER BY run_at ASC
                """,
                (now_iso,),
            ).fetchall()
        return [dict(r) for r in rows]

    def log_episode(self, title: str, detail: str = "", outcome: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes(title, detail, outcome, created_at) VALUES (?, ?, ?, ?)",
                (title, detail, outcome, self._now()),
            )
            self._conn.commit()
            eid = int(cur.lastrowid)
        # Also store as a searchable memory of kind episode
        try:
            self.remember(
                f"{title}: {detail} [{outcome}]".strip(": "),
                kind="episode",
                tags="episode",
            )
        except Exception:
            pass
        return eid

    def list_episodes(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- prompt assembly & auto-extract ---

    def build_memory_context(self, user_text: str) -> str:
        """Relevant memories + latest summary for injection into the system prompt."""
        parts: list[str] = []
        summary = self.get_latest_summary()
        if summary:
            parts.append("## Conversation summary so far\n" + summary.strip())
        items = self.recall(user_text, limit=MAX_RECALL_ITEMS, use_embeddings=False)
        if not items:
            # Always surface a few durable preferences/facts
            items = self.list_memories(limit=8, kinds=["fact", "preference"])
        if items:
            lines = [f"- [{m.kind}] {m.content}" for m in items]
            parts.append("## Long-term memory (use when relevant)\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def auto_extract(self, user_text: str, assistant_text: str = "") -> list[MemoryItem]:
        """Heuristic extraction of durable facts/preferences from a turn.

        Looks for explicit remember cues and preference patterns. A smarter
        LLM pass can call remember() itself; this catches the obvious cases
        even when the model forgets to.
        """
        saved: list[MemoryItem] = []
        text = f"{user_text}\n{assistant_text}"
        patterns = [
            (r"(?i)\bremember(?:\s+that)?\s+(.+)", "fact"),
            (r"(?i)\bmy name is\s+([A-Z][\w'-]+)", "fact"),
            (r"(?i)\bi(?:'m| am) (?:called|named)\s+([A-Z][\w'-]+)", "fact"),
            (r"(?i)\bi (?:prefer|like|love|hate|dislike)\s+(.+)", "preference"),
            (r"(?i)\balways\s+(.+)", "preference"),
            (r"(?i)\bnever\s+(.+)", "preference"),
            (r"(?i)\bcall me\s+([A-Z][\w'-]+)", "preference"),
        ]
        for pat, kind in patterns:
            for m in re.finditer(pat, user_text):
                content = m.group(0).strip()
                # Normalize short captures into readable memories
                if kind == "fact" and m.lastindex and "name" in m.group(0).lower():
                    content = f"User's name is {m.group(1).strip()}"
                elif kind == "preference" and m.lastindex and m.group(0).lower().startswith("call me"):
                    content = f"User prefers to be called {m.group(1).strip()}"
                elif len(content) > 240:
                    content = content[:240].rstrip() + "…"
                try:
                    saved.append(self.remember(content, kind=kind, tags="auto"))
                except Exception as exc:
                    logger.debug("auto_extract skip: %s", exc)
        # Dedup by content already handled by UNIQUE constraint
        return saved

    def tool_specs(self) -> list[dict]:
        """Gemini function-calling declarations for memory tools."""
        return [
            {
                "name": "remember",
                "description": (
                    "Store a durable fact, preference, or note in long-term memory. "
                    "Use for anything that should persist across sessions "
                    "(name, preferences, project details, decisions)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact or preference to remember, written clearly.",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["fact", "preference", "note", "episode"],
                            "description": "Category of memory.",
                        },
                        "tags": {
                            "type": "string",
                            "description": "Optional comma-separated tags.",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "recall",
                "description": (
                    "Search long-term memory for relevant facts, preferences, notes, "
                    "or past episodes. Call this when the user asks what you remember, "
                    "or before acting on something that may have prior context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look up.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 8).",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "forget",
                "description": (
                    "Delete a memory by id, or all memories matching a query. "
                    "Confirm with the user before forgetting important facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "Exact memory id to delete.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Delete memories matching this search.",
                        },
                    },
                },
            },
            {
                "name": "log_episode",
                "description": (
                    "Record a completed task or notable episode for future recall "
                    "(what happened and the outcome)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "detail": {"type": "string"},
                        "outcome": {"type": "string"},
                    },
                    "required": ["title"],
                },
            },
        ]

    def dispatch_tool(self, name: str, args: dict) -> dict:
        """Execute a memory tool call. Returns a JSON-serializable result."""
        try:
            if name == "remember":
                item = self.remember(
                    args.get("content", ""),
                    kind=args.get("kind", "note"),
                    tags=args.get("tags", ""),
                )
                return {"ok": True, "memory": item.to_dict()}
            if name == "recall":
                items = self.recall(
                    args.get("query", ""),
                    limit=int(args.get("limit") or 8),
                )
                return {"ok": True, "results": [i.to_dict() for i in items]}
            if name == "forget":
                mid = args.get("memory_id")
                n = self.forget(
                    memory_id=int(mid) if mid is not None else None,
                    query=args.get("query"),
                )
                return {"ok": True, "deleted": n}
            if name == "log_episode":
                eid = self.log_episode(
                    args.get("title", "episode"),
                    detail=args.get("detail", ""),
                    outcome=args.get("outcome", ""),
                )
                return {"ok": True, "episode_id": eid}
            return {"ok": False, "error": f"unknown memory tool: {name}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
