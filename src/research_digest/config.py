"""Settings loaded from environment (prefix DIGEST_) or .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default: arxiv-graph stores its DB in ~/.arxiv-graph/data/arxiv_graph.db
_DEFAULT_DB_PATH = str(Path.home() / ".arxiv-graph" / "data" / "arxiv_graph.db")
_DEFAULT_STATE_DIR = str(Path.home() / ".research-digest")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIGEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    arxiv_db_path: str = Field(
        default=_DEFAULT_DB_PATH,
        description="Path to arxiv-graph sqlite DB",
    )
    slack_webhook_url: str | None = Field(default=None, description="Optional Slack webhook URL")
    top_n: int = Field(default=5, ge=1, le=50)
    model: str = Field(default="claude-sonnet-4-6")

    # State directory for feedback.db, embedding cache, and deep-read cache.
    state_dir: str = Field(default=_DEFAULT_STATE_DIR)

    # Voyage AI for embeddings; if empty, rank.py falls back to a local TF-IDF.
    voyage_api_key: str = Field(default="", description="Voyage AI API key (optional)")
    voyage_model: str = Field(default="voyage-3")


def get_settings() -> Settings:
    return Settings()


def state_paths(state_dir: str) -> dict[str, Path]:
    """Resolve standard sqlite/cache paths under the state dir, creating it if needed."""
    root = Path(state_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    deepread_dir = root / "deepread_cache"
    deepread_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "feedback_db": root / "feedback.db",
        "deepread_cache": deepread_dir,
    }
