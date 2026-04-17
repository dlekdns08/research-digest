"""Click CLI entry point."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markdown import Markdown

from . import feedback
from .cluster import cluster_and_dedup
from .config import get_settings
from .db import ArxivDBNotFoundError, fetch_top_papers
from .deepread import deep_read
from .deliver import post_to_slack, print_to_console
from .rank import rank_personalized
from .render import PaperWithSummary, render_markdown, render_slack_blocks
from .summarize import summarize_paper


@click.group()
def cli() -> None:
    """research-digest: daily arxiv paper digest."""


@cli.command()
@click.option("--top", "top_n", type=int, default=None, help="Number of papers to include.")
@click.option("--since-days", type=int, default=1, help="Only papers published within N days.")
@click.option("--slack/--no-slack", default=False, help="Post to Slack webhook.")
@click.option(
    "--personalize/--no-personalize",
    default=None,
    help="Enable personalized re-ranking. Default: on if ≥5 liked papers exist.",
)
@click.option(
    "--include-seen/--skip-seen",
    default=False,
    help="Include papers already shown in previous runs (default: skip).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip LLM / Voyage / delivery; just list the papers that would be summarized.",
)
def run(
    top_n: int | None,
    since_days: int,
    slack: bool,
    personalize: bool | None,
    include_seen: bool,
    dry_run: bool,
) -> None:
    """Fetch top papers, rank, dedup, summarize, and deliver."""
    settings = get_settings()
    console = Console(stderr=True)
    n = top_n or settings.top_n

    # Pull 3N candidates so personalization + clustering has room to reorder/dedup.
    pool_size = max(n * 3, n)

    try:
        pool = fetch_top_papers(settings.arxiv_db_path, n=pool_size, since_days=since_days)
    except ArxivDBNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(2)

    # Filter out already-seen papers unless the user asked to include them.
    if not include_seen:
        seen = feedback.seen_set()
        pool = [p for p in pool if p.arxiv_id not in seen]

    if not pool:
        console.print("[yellow]No papers found for the selected window.[/yellow]")
        print_to_console(render_markdown([]))
        return

    # Decide whether to personalize.
    if personalize is None:
        personalize = feedback.liked_count() >= 5

    if dry_run:
        # In dry-run we do NOT call Voyage or Claude. Pick by importance only.
        pool.sort(key=lambda p: p.importance_score, reverse=True)
        picks = pool[:n]
        console.print(
            f"[green]Dry run — {len(picks)} papers would be summarized "
            f"(pool {len(pool)}, personalize={personalize}):[/green]"
        )
        for i, p in enumerate(picks, 1):
            console.print(
                f"  {i}. [{p.arxiv_id}] score={p.importance_score:.2f} {p.title[:100]}"
            )
        return

    if not settings.anthropic_api_key:
        console.print(
            "[red]error:[/red] DIGEST_ANTHROPIC_API_KEY is not set. "
            "Use --dry-run to skip summarization."
        )
        sys.exit(2)

    # 1. Rank (embeds each candidate and blends with importance).
    scored = rank_personalized(pool, settings=settings, personalize=personalize)

    # 2. Dedup by cosine clustering.
    clustered = cluster_and_dedup(scored, threshold=0.25)

    # 3. Take the top-N heads.
    heads = clustered[:n]

    # 4. Summarize each.
    items: list[PaperWithSummary] = []
    for i, head in enumerate(heads, 1):
        p = head.scored.paper
        console.print(f"[dim]summarizing {i}/{len(heads)}: {p.arxiv_id}[/dim]")
        try:
            summary = summarize_paper(
                title=p.title,
                abstract=p.abstract,
                model=settings.model,
                api_key=settings.anthropic_api_key,
            )
        except Exception as e:  # noqa: BLE001 - surface any SDK error gracefully
            console.print(f"[yellow]warn:[/yellow] summarize failed for {p.arxiv_id}: {e}")
            summary = "_(요약 실패)_"
        items.append(
            PaperWithSummary(
                paper=p,
                summary=summary,
                related_count=head.related_count,
                related_ids=head.related_ids,
                personalization=head.scored.personalization if personalize else None,
            )
        )

    md = render_markdown(items)
    print_to_console(md)

    # Mark the papers we actually surfaced (heads + their merged siblings).
    surfaced_ids: list[str] = []
    for h in heads:
        surfaced_ids.append(h.scored.paper.arxiv_id)
        surfaced_ids.extend(h.related_ids)
    feedback.mark_seen(surfaced_ids)

    if slack:
        if not settings.slack_webhook_url:
            console.print("[red]error:[/red] --slack passed but DIGEST_SLACK_WEBHOOK_URL unset.")
            sys.exit(2)
        try:
            post_to_slack(render_slack_blocks(items), settings.slack_webhook_url)
            console.print("[green]Posted to Slack.[/green]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error:[/red] slack post failed: {e}")
            sys.exit(1)


@cli.command()
@click.argument("arxiv_id")
def like(arxiv_id: str) -> None:
    """Record a +1 signal for this paper (improves future personalization)."""
    feedback.record(arxiv_id, signal=+1)
    click.echo(f"liked {arxiv_id}")


@cli.command()
@click.argument("arxiv_id")
def skip(arxiv_id: str) -> None:
    """Record a -1 signal for this paper."""
    feedback.record(arxiv_id, signal=-1)
    click.echo(f"skipped {arxiv_id}")


@cli.command("feedback-stats")
def feedback_stats() -> None:
    """Show like/skip/seen/embedded counts from the local feedback DB."""
    s = feedback.stats()
    click.echo(
        f"likes={s['likes']}  skips={s['skips']}  seen={s['seen']}  embedded={s['embedded']}"
    )


@cli.command()
@click.argument("arxiv_id")
@click.option("--refresh", is_flag=True, default=False, help="Ignore cache and re-fetch the PDF.")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Skip network + LLM calls; print what would be done.",
)
def deepread(arxiv_id: str, refresh: bool, dry_run: bool) -> None:
    """Fetch the PDF, extract it, and produce a structured Korean summary."""
    settings = get_settings()
    console = Console(stderr=True)
    if not dry_run and not settings.anthropic_api_key:
        console.print(
            "[red]error:[/red] DIGEST_ANTHROPIC_API_KEY is not set. "
            "Use --dry-run to preview."
        )
        sys.exit(2)
    try:
        out = deep_read(arxiv_id, settings=settings, refresh=refresh, dry_run=dry_run)
    except ArxivDBNotFoundError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(2)
    except LookupError as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]error:[/red] deepread failed: {e}")
        sys.exit(1)
    Console().print(Markdown(out))


if __name__ == "__main__":
    cli()
