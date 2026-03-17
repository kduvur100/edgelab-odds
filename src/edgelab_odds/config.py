"""
edgelab_odds.config
===================
Central settings object loaded from environment variables / .env file.

Usage:
    from edgelab_odds.config import settings

    print(settings.db_path)
    print(settings.ufc_csv_path)
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = two levels up from this file (src/edgelab_odds/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration for edgelab-odds.

    Values are read from environment variables or a .env file at the
    project root.  Defaults are sensible for local development.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    db_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "edgelab.duckdb",
        description="Path to the DuckDB database file.",
    )

    # ── Raw data ──────────────────────────────────────────────────────────
    ufc_csv_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "ufc-master.csv",
        description="Path to the UFC master CSV dataset.",
    )

    # ── Odds API ──────────────────────────────────────────────────────────
    odds_api_key: str = Field(
        default="",
        description="API key for the-odds-api.com (optional, for live odds).",
    )
    odds_api_base_url: str = Field(
        default="https://api.the-odds-api.com/v4",
        description="Base URL for the odds API.",
    )

    # ── Model artifacts ───────────────────────────────────────────────────
    model_dir: Path = Field(
        default=_PROJECT_ROOT / "models",
        description="Directory where trained model pickles are saved.",
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    random_seed: int = Field(default=42, description="Global random seed.")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Python logging level.")

    # ── Derived helpers (not env vars) ────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    def ensure_dirs(self) -> None:
        """Create data/ and models/ directories if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere
settings = Settings()
