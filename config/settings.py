"""Global application settings."""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Path = Field(default=None)
    models_dir: Path = Field(default=None)
    logs_dir: Path = Field(default=None)

    # API Keys (optional - free tiers work without for basic usage)
    coingecko_api_key: str = Field(default="", env="COINGECKO_API_KEY")

    # Database
    database_url: str = Field(default="sqlite:///data/trading.db")

    # Logging
    log_level: str = Field(default="INFO")

    # Trading defaults
    default_symbols: list[str] = Field(default=["bitcoin", "ethereum"])
    default_timeframe: str = Field(default="daily")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set derived paths
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models" / "saved"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
