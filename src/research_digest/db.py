"""Read-only access to the arxiv-graph sqlite DB."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class PaperRow:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str | None
    importance_score: float
    citation_count: int
    published_at: str | None
    primary_category: str | None


class ArxivDBNotFoundError(FileNotFoundError):
    """Raised when the arxiv-graph DB is missing."""


def fetch_top_papers(
    db_path: str | Path,
    n: int,
    since_days: int | None = 1,
) -> list[PaperRow]:
    """Return top-N papers ordered by importance_score desc.

    If `since_days` is provided, only include papers with
    published_at >= (now - since_days). Pass `None` (or 0) to disable.
    """
    path = Path(db_path)
    if not path.exists():
        raise ArxivDBNotFoundError(
            f"arxiv-graph DB not found at {path}. "
            f"Run arxiv-graph first, or set DIGEST_ARXIV_DB_PATH."
        )

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        params: list = []
        where = ""
        if since_days and since_days > 0:
            cutoff = (datetime.utcnow() - timedelta(days=since_days)).isoformat()
            where = "WHERE published_at >= ?"
            params.append(cutoff)
        sql = f"""
            SELECT arxiv_id, title, abstract, pdf_url, importance_score,
                   citation_count, published_at, primary_category
            FROM papers
            {where}
            ORDER BY importance_score DESC, citation_count DESC
            LIMIT ?
        """
        params.append(n)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return [
        PaperRow(
            arxiv_id=r["arxiv_id"],
            title=r["title"] or "",
            abstract=r["abstract"] or "",
            pdf_url=r["pdf_url"],
            importance_score=float(r["importance_score"] or 0.0),
            citation_count=int(r["citation_count"] or 0),
            published_at=r["published_at"],
            primary_category=r["primary_category"],
        )
        for r in rows
    ]
