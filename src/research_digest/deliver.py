"""Delivery sinks: Slack webhook and rich console."""

from __future__ import annotations

import httpx
from rich.console import Console
from rich.markdown import Markdown


def post_to_slack(blocks: list[dict], webhook_url: str, timeout: float = 10.0) -> None:
    """POST Block Kit payload to a Slack incoming webhook."""
    if not webhook_url:
        raise ValueError("Slack webhook URL is empty.")
    payload = {"blocks": blocks}
    resp = httpx.post(webhook_url, json=payload, timeout=timeout)
    resp.raise_for_status()


def print_to_console(markdown: str) -> None:
    """Pretty-print markdown to the terminal."""
    console = Console()
    console.print(Markdown(markdown))
