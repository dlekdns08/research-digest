"""Local sqlite store for user feedback, seen-history, and embedding cache.

Tables (all created lazily with CREATE TABLE IF NOT EXISTS):

    feedback(arxiv_id TEXT PRIMARY KEY, signal INTEGER, ts INTEGER)
    seen(arxiv_id TEXT PRIMARY KEY, ts INTEGER)
    embeddings(arxiv_id TEXT PRIMARY KEY, vector BLOB)

Migrations are additive only — never drop or rename columns.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .config import get_settings, state_paths


def _db_path() -> Path:
    return state_paths(get_settings().state_dir)["feedback_db"]


def _connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or _db_path()
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            arxiv_id TEXT PRIMARY KEY,
            signal   INTEGER NOT NULL,
            ts       INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS seen (
            arxiv_id TEXT PRIMARY KEY,
            ts       INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            arxiv_id TEXT PRIMARY KEY,
            vector   BLOB NOT NULL
        );
        """
    )
    conn.commit()


# ---------- feedback ----------


def record(arxiv_id: str, signal: int) -> None:
    """Insert or replace a like(+1)/skip(-1) signal for this paper."""
    if signal not in (1, -1):
        raise ValueError("signal must be +1 (like) or -1 (skip)")
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO feedback(arxiv_id, signal, ts) VALUES(?, ?, ?)",
            (arxiv_id, signal, int(time.time())),
        )
        conn.commit()


def recent_liked(limit: int = 20) -> list[str]:
    """Return the most recently liked arxiv_ids (signal=+1), newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT arxiv_id FROM feedback WHERE signal = 1 ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [r["arxiv_id"] for r in rows]


def liked_count() -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM feedback WHERE signal = 1"
        ).fetchone()
    return int(row["c"] or 0)


def stats() -> dict:
    with _connect() as conn:
        likes = conn.execute(
            "SELECT COUNT(*) AS c FROM feedback WHERE signal = 1"
        ).fetchone()["c"]
        skips = conn.execute(
            "SELECT COUNT(*) AS c FROM feedback WHERE signal = -1"
        ).fetchone()["c"]
        seen = conn.execute("SELECT COUNT(*) AS c FROM seen").fetchone()["c"]
        embedded = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
    return {
        "likes": int(likes or 0),
        "skips": int(skips or 0),
        "seen": int(seen or 0),
        "embedded": int(embedded or 0),
    }


# ---------- seen ----------


def mark_seen(arxiv_ids: list[str]) -> None:
    if not arxiv_ids:
        return
    now = int(time.time())
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO seen(arxiv_id, ts) VALUES(?, ?)",
            [(aid, now) for aid in arxiv_ids],
        )
        conn.commit()


def seen_set() -> set[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT arxiv_id FROM seen").fetchall()
    return {r["arxiv_id"] for r in rows}


# ---------- embedding cache ----------


def get_embedding(arxiv_id: str) -> bytes | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    return bytes(row["vector"]) if row else None


def put_embedding(arxiv_id: str, vector: bytes) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(arxiv_id, vector) VALUES(?, ?)",
            (arxiv_id, vector),
        )
        conn.commit()


def get_embeddings(arxiv_ids: list[str]) -> dict[str, bytes]:
    if not arxiv_ids:
        return {}
    with _connect() as conn:
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = conn.execute(
            f"SELECT arxiv_id, vector FROM embeddings WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
    return {r["arxiv_id"]: bytes(r["vector"]) for r in rows}
