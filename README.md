# research-digest

매일 아침 [arxiv-graph](../arxiv-graph/)가 수집·랭킹한 오늘의 top-N 논문을 Claude로 요약하여 콘솔 또는 Slack으로 전달하는 CLI.

개인화 랭킹(Voyage 임베딩), PDF 딥리드, 중복 제거 클러스터링까지 포함합니다.

## Install

```bash
cd research-digest
uv sync      # or: pip install -e .
cp .env.example .env   # then fill in DIGEST_ANTHROPIC_API_KEY
```

## Run

```bash
# Preview which papers would be picked (no LLM/Voyage calls)
research-digest run --dry-run

# Print today's digest to the console (auto-personalizes after ≥5 likes)
research-digest run --top 5 --since-days 1

# Post to Slack as well
research-digest run --slack

# Send the digest as an HTML email (top 10)
research-digest run --top 10 --email

# Force personalization on/off
research-digest run --personalize
research-digest run --no-personalize

# Include papers already shown in previous runs
research-digest run --include-seen
```

### Feedback

`run` automatically marks surfaced papers as "seen" so they won't be repeated.
Record likes/skips to personalize future digests:

```bash
research-digest like 2404.12345
research-digest skip 2404.98765
research-digest feedback-stats
```

Personalized ranking blends a cosine similarity to the centroid of your
recently-liked papers (weight 0.6) with the normalized arxiv-graph
importance score (weight 0.4). With `DIGEST_VOYAGE_API_KEY` set, embeddings
come from `voyage-3`; otherwise a small local hashed TF-IDF is used.

### Deep read

Fetch a paper's PDF, extract the text, and get a structured Korean summary
(문제 정의 / 접근법 / 핵심 수식·알고리즘 / 실험 결과 / 한계):

```bash
research-digest deepread 2404.12345
research-digest deepread 2404.12345 --refresh
research-digest deepread 2404.12345 --dry-run
```

Output is cached at `~/.research-digest/deepread_cache/<arxiv_id>.md`.

## Environment variables

All env vars use the `DIGEST_` prefix and can live in `.env`:

| Var | Default | Description |
| --- | --- | --- |
| `DIGEST_ANTHROPIC_API_KEY` | — | Anthropic API key (required unless `--dry-run`) |
| `DIGEST_ARXIV_DB_PATH` | `~/.arxiv-graph/data/arxiv_graph.db` | Path to the sibling arxiv-graph sqlite DB (read-only) |
| `DIGEST_SLACK_WEBHOOK_URL` | — | Optional Slack incoming webhook URL |
| `DIGEST_TOP_N` | `5` | Number of papers in each digest |
| `DIGEST_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `DIGEST_VOYAGE_API_KEY` | — | Optional Voyage AI key for embeddings |
| `DIGEST_VOYAGE_MODEL` | `voyage-3` | Voyage embedding model |
| `DIGEST_STATE_DIR` | `~/.research-digest` | Dir for feedback.db + deepread cache |
| `DIGEST_EMAIL_FROM` | `dlekdns08@gmail.com` | Sender Gmail (must have 2FA + app password) |
| `DIGEST_EMAIL_TO` | `dlekdns07@naver.com` | Recipient address |
| `DIGEST_SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `DIGEST_SMTP_PORT` | `465` | `465` (SSL) or `587` (STARTTLS) |
| `DIGEST_SMTP_USER` | same as `EMAIL_FROM` | SMTP login user |
| `DIGEST_SMTP_PASSWORD` | — | Gmail 16-char **app password** (required for `--email`) |

## Daily at 07:00 KST (GitHub Actions, self-hosted runner)

`.github/workflows/daily-digest.yml` schedules the digest every day at
**22:00 UTC = 07:00 KST** on a self-hosted runner. Assumes arxiv-graph's
`daily_crawl.yml` runs on the same server (different runner is fine) so the
sqlite DB at `/root/.arxiv-graph/data/arxiv_graph.db` is fresh by morning.

Required GitHub Secrets (*Settings → Secrets and variables → Actions*):

| Secret | Purpose |
| --- | --- |
| `DIGEST_ANTHROPIC_API_KEY` | Claude API key |
| `DIGEST_SMTP_PASSWORD` | Gmail 16-char app password |
| `DIGEST_VOYAGE_API_KEY` | *(optional)* Voyage embeddings for personalization |

Other config (sender/recipient, DB path) is hard-coded in the workflow `env`
block — edit the YAML directly if you need to change them.

Manual trigger / testing:

- Push to `main` (touching `src/`, `pyproject.toml`, or the workflow itself)
- *Actions → Daily Research Digest → Run workflow* (`workflow_dispatch`)

If a run fails, an issue is auto-filed with a link to the run log.

**Requirement:** the digest runner must have **read access** to
`/root/.arxiv-graph/data/arxiv_graph.db`. Easiest if both runners run as
`root`; otherwise `chmod o+r` the DB file or run digest under a user in
the same group.

## What it does

1. Opens arxiv-graph's sqlite DB read-only and pulls a 3N candidate pool.
2. Skips already-seen papers (unless `--include-seen`).
3. Embeds each candidate (Voyage or local TF-IDF fallback) and re-ranks by
   `0.6 * cosine(사용자 선호 중심) + 0.4 * 정규화 importance`.
4. Agglomerative cosine clustering (threshold 0.25) merges near-duplicates;
   the top paper per cluster keeps a "관련: 외 N편" tag.
5. Asks Claude for 2-3문장 한국어 요약 + "왜 볼만한지" 한 줄.
6. Renders the digest as Markdown for the console (via `rich`) and/or Slack
   Block Kit blocks; marks surfaced ids as seen.
