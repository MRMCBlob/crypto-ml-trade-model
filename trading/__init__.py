"""Trading module for paper trading and strategy execution."""
from .orchestrator import MultiCoinTradingOrchestrator, CoinSignal

__all__ = ["MultiCoinTradingOrchestrator", "CoinSignal"]
