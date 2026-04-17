"""On-demand PDF deep-read: download, extract, summarize in Korean."""

from __future__ import annotations

import io
from pathlib import Path

import httpx
from anthropic import Anthropic
from pypdf import PdfReader

from .config import Settings, state_paths
from .db import ArxivDBNotFoundError, PaperRow

# Rough char budget for ~30k tokens (≈4 chars per token average).
_MAX_CHARS = 120_000

_SYSTEM_PROMPT = """당신은 AI/ML 논문을 깊이 있게 읽고 구조화된 한국어 요약을 작성하는 리서치 어시스턴트입니다.

다음 5개 섹션을 마크다운 헤더(##)로 구분해 순서대로 작성하세요:

## 문제 정의
해결하려는 문제와 맥락을 2-4문장으로 설명.

## 접근법
제안 방법의 핵심 아이디어와 구조를 3-6문장으로 설명.

## 핵심 수식·알고리즘
논문에 등장하는 주요 수식이나 알고리즘 단계 2-4개를 불릿으로 정리. 수식은 LaTeX 문법을 그대로 옮겨도 됩니다.

## 실험 결과
주요 벤치마크, 수치, 비교 대상 베이스라인을 불릿으로 정리.

## 한계
저자가 언급했거나 본문에서 드러나는 한계/리스크 2-4개.

규칙:
- 한국어로 작성, 전문 용어는 원어 병기 허용.
- 본문에 없는 내용을 지어내지 말 것.
- 서론/꼬리말 없이 헤더부터 바로 시작.
"""


def _pdf_url_for(arxiv_id: str, stored: str | None) -> str:
    if stored:
        return stored
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _fetch_pdf_bytes(url: str, timeout: float = 60.0) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - PDFs routinely have broken pages
            continue
    return "\n\n".join(parts).strip()


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # Keep head and tail — often both intro and conclusions are useful.
    head = text[: int(max_chars * 0.75)]
    tail = text[-int(max_chars * 0.25) :]
    return f"{head}\n\n[... 중략 ...]\n\n{tail}"


def _cache_path(settings: Settings, arxiv_id: str) -> Path:
    return state_paths(settings.state_dir)["deepread_cache"] / f"{arxiv_id}.md"


def _lookup_paper(settings: Settings, arxiv_id: str) -> PaperRow | None:
    """Find a single paper row by id using the read-only arxiv-graph DB."""
    import sqlite3  # local import keeps module load cheap

    path = Path(settings.arxiv_db_path)
    if not path.exists():
        raise ArxivDBNotFoundError(f"arxiv-graph DB not found at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT arxiv_id, title, abstract, pdf_url, importance_score,
                   citation_count, published_at, primary_category
            FROM papers WHERE arxiv_id = ?
            """,
            (arxiv_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return PaperRow(
        arxiv_id=row["arxiv_id"],
        title=row["title"] or "",
        abstract=row["abstract"] or "",
        pdf_url=row["pdf_url"],
        importance_score=float(row["importance_score"] or 0.0),
        citation_count=int(row["citation_count"] or 0),
        published_at=row["published_at"],
        primary_category=row["primary_category"],
    )


def deep_read(
    arxiv_id: str,
    settings: Settings,
    refresh: bool = False,
    dry_run: bool = False,
) -> str:
    """Return a structured Korean deep-read of the paper.

    Uses a filesystem cache at `<state_dir>/deepread_cache/<arxiv_id>.md`.
    """
    cache_file = _cache_path(settings, arxiv_id)
    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")

    paper = _lookup_paper(settings, arxiv_id)
    if paper is None:
        raise LookupError(f"arxiv_id {arxiv_id} not in arxiv-graph DB")

    url = _pdf_url_for(arxiv_id, paper.pdf_url)

    if dry_run:
        return (
            f"# Deep read (dry run): {paper.title}\n\n"
            f"Would fetch PDF from: {url}\n"
            f"Would call Claude model: {settings.model}\n"
        )

    pdf_bytes = _fetch_pdf_bytes(url)
    text = _truncate(_extract_text(pdf_bytes))
    if not text:
        raise RuntimeError(f"Could not extract any text from PDF at {url}")

    client = (
        Anthropic(api_key=settings.anthropic_api_key)
        if settings.anthropic_api_key
        else Anthropic()
    )
    user_content = (
        f"제목(title): {paper.title}\n\n"
        f"초록(abstract):\n{paper.abstract.strip()}\n\n"
        f"본문(body, 일부 발췌):\n{text}"
    )
    resp = client.messages.create(
        model=settings.model,
        max_tokens=2000,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
    body = "\n".join(parts).strip()

    header = f"# {paper.title}\n\n`{arxiv_id}` · [pdf]({url})\n\n"
    output = header + body + "\n"
    cache_file.write_text(output, encoding="utf-8")
    return output
