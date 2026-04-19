"""Delivery sinks: Slack webhook, email (SMTP), and rich console."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

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


def send_email(
    *,
    subject: str,
    html_body: str,
    text_body: str,
    sender: str,
    recipient: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    timeout: float = 15.0,
) -> None:
    """Send an HTML+text email via SMTP (TLS).

    For Gmail: host=smtp.gmail.com, port=465 (SSL) or 587 (STARTTLS),
    user=<your@gmail>, password=<16-char app password — NOT your normal
    account password>. Requires 2FA enabled on the sender account.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout) as s:
            s.login(smtp_user, smtp_password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as s:
            s.starttls()
            s.login(smtp_user, smtp_password)
            s.send_message(msg)
