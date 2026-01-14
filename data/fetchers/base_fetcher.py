"""Abstract base class for data fetchers."""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.

        Args:
            symbol: The cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: datetime
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Return list of available trading symbols.

        Returns:
            List of symbol strings
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: The cryptocurrency symbol

        Returns:
            Current price in USD
        """
        pass
