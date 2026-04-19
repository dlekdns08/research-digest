"""Render the digest as Markdown (console) and Slack Block Kit blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .db import PaperRow


@dataclass
class PaperWithSummary:
    paper: PaperRow
    summary: str
    # Optional cluster metadata: number of related (deduped) papers and
    # a personalization similarity score in [-1, 1].
    related_count: int = 0
    related_ids: list[str] = field(default_factory=list)
    personalization: float | None = None


def _arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _meta_line(p: PaperRow, it: PaperWithSummary) -> str:
    parts = [
        f"`{p.arxiv_id}`",
        f"score {p.importance_score:.2f}",
        f"citations {p.citation_count}",
        p.primary_category or "-",
    ]
    if it.personalization is not None:
        parts.append(f"sim {it.personalization:+.2f}")
    if it.related_count > 0:
        parts.append(f"관련: 외 {it.related_count}편")
    return " · ".join(parts)


def render_markdown(items: list[PaperWithSummary]) -> str:
    if not items:
        return "# Today's Research Digest\n\n_No papers found for the selected window._\n"

    lines: list[str] = ["# Today's Research Digest", ""]
    for i, it in enumerate(items, 1):
        p = it.paper
        link = p.pdf_url or _arxiv_url(p.arxiv_id)
        lines.append(f"## {i}. {p.title}")
        lines.append(_meta_line(p, it))
        lines.append("")
        lines.append(it.summary.strip())
        lines.append("")
        if it.related_ids:
            rel = ", ".join(f"`{aid}`" for aid in it.related_ids)
            lines.append(f"_관련 논문: {rel}_")
            lines.append("")
        lines.append(f"[arxiv/pdf]({link})")
        lines.append("")
    return "\n".join(lines)


def render_email_html(items: list[PaperWithSummary]) -> str:
    """Render the digest as a simple inline-styled HTML email body."""
    import html as _html

    if not items:
        return (
            "<h1>Today's Research Digest</h1>"
            "<p><em>No papers found for the selected window.</em></p>"
        )

    parts: list[str] = [
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:720px;margin:0 auto;color:#1a1a1a;line-height:1.55;\">",
        "<h1 style=\"border-bottom:2px solid #eee;padding-bottom:8px;\">"
        "Today's Research Digest</h1>",
    ]
    for i, it in enumerate(items, 1):
        p = it.paper
        link = p.pdf_url or _arxiv_url(p.arxiv_id)
        title = _html.escape(p.title)
        summary_html = _html.escape(it.summary.strip()).replace("\n", "<br>")
        meta = _html.escape(_meta_line(p, it))

        parts.append(
            f"<h2 style=\"margin-top:28px;font-size:18px;\">"
            f"{i}. <a href=\"{_html.escape(link)}\" "
            f"style=\"color:#0b5ed7;text-decoration:none;\">{title}</a></h2>"
        )
        parts.append(
            f"<div style=\"color:#666;font-size:12px;font-family:ui-monospace,monospace;"
            f"margin-bottom:8px;\">{meta}</div>"
        )
        parts.append(f"<div>{summary_html}</div>")
        if it.related_ids:
            rel = ", ".join(_html.escape(aid) for aid in it.related_ids)
            parts.append(
                f"<div style=\"color:#888;font-size:12px;margin-top:6px;\">"
                f"관련 논문: {rel}</div>"
            )
        parts.append(
            f"<div style=\"margin-top:8px;font-size:12px;\">"
            f"<a href=\"{_html.escape(link)}\" style=\"color:#0b5ed7;\">arxiv/pdf →</a></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_slack_blocks(items: list[PaperWithSummary]) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Today's Research Digest"},
        }
    ]

    if not items:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "_No papers found for the selected window._"},
            }
        )
        return blocks

    for i, it in enumerate(items, 1):
        p = it.paper
        link = p.pdf_url or _arxiv_url(p.arxiv_id)
        header_text = f"*{i}. <{link}|{p.title}>*"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": header_text}})
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": _meta_line(p, it)}],
            }
        )
        blocks.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": it.summary.strip()}}
        )
        if it.related_ids:
            rel = ", ".join(f"`{aid}`" for aid in it.related_ids)
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"관련 논문: {rel}"}],
                }
            )
        blocks.append({"type": "divider"})

    return blocks
