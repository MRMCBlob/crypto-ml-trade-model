"""Global application settings."""
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_dir: Optional[Path] = Field(default=None)
    models_dir: Optional[Path] = Field(default=None)
    logs_dir: Optional[Path] = Field(default=None, alias="LOG_DIR")

    # API Keys (optional - free tiers work without for basic usage)
    coingecko_api_key: str = Field(default="", alias="COINGECKO_API_KEY")

    # Database
    database_path: str = Field(default="storage/trading.db", alias="DATABASE_PATH")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Trading mode
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")

    # Trading configuration (can override TradingConfig defaults)
    initial_capital: float = Field(default=10000.0, alias="INITIAL_CAPITAL")
    max_position_pct: float = Field(default=0.20, alias="MAX_POSITION_PCT")
    max_total_exposure: float = Field(default=0.80, alias="MAX_TOTAL_EXPOSURE")
    risk_per_trade_pct: float = Field(default=0.02, alias="RISK_PER_TRADE_PCT")
    max_drawdown_pct: float = Field(default=0.15, alias="MAX_DRAWDOWN_PCT")
    max_daily_loss_pct: float = Field(default=0.05, alias="MAX_DAILY_LOSS_PCT")

    # Model configuration
    prediction_horizon: int = Field(default=7, alias="PREDICTION_HORIZON")
    sequence_length: int = Field(default=30, alias="SEQUENCE_LENGTH")
    validation_split: float = Field(default=0.2, alias="VALIDATION_SPLIT")

    # Category allocation overrides (optional)
    max_allocation_large_cap: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_LARGE_CAP")
    max_allocation_altcoin: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_ALTCOIN")
    max_allocation_defi: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_DEFI")
    max_allocation_layer2: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_LAYER2")
    max_allocation_memecoin: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_MEMECOIN")
    max_allocation_ai: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_AI")
    max_allocation_gaming: Optional[float] = Field(default=None, alias="MAX_ALLOCATION_GAMING")

    # Trading defaults
    default_symbols: list[str] = Field(default=["bitcoin", "ethereum"])
    default_timeframe: str = Field(default="daily")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True

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
