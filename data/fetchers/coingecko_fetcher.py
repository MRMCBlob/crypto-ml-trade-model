"""CoinGecko API data fetcher."""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .base_fetcher import BaseFetcher
from utils.decorators import rate_limit, retry_on_failure
from utils.logger import get_logger

logger = get_logger(__name__)


# Common symbol mappings (ticker -> CoinGecko ID)
SYMBOL_MAPPING = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LTC": "litecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XLM": "stellar",
}


class CoinGeckoFetcher(BaseFetcher):
    """
    CoinGecko API data fetcher.

    Free tier: 30 calls/min (Demo API key), 10-30 calls/min (no key)
    Provides daily OHLC data up to 365 days with granularity.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko fetcher.

        Args:
            api_key: Optional CoinGecko Demo API key for higher rate limits
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoMLTrader/1.0"
        })
        if api_key:
            self.session.headers["x-cg-demo-api-key"] = api_key

    def _symbol_to_id(self, symbol: str) -> str:
        """Convert ticker symbol to CoinGecko ID."""
        symbol_upper = symbol.upper()
        if symbol_upper in SYMBOL_MAPPING:
            return SYMBOL_MAPPING[symbol_upper]
        # Assume it's already a CoinGecko ID
        return symbol.lower()

    @rate_limit(calls=25, period=60)
    @retry_on_failure(max_retries=3, delay=2.0)
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make API request with rate limiting and retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from CoinGecko.

        Note: CoinGecko's OHLC endpoint provides candles, not volume.
        We fetch market_chart for price + volume data instead.

        Args:
            symbol: Cryptocurrency symbol or CoinGecko ID
            start_date: Start date (YYYY-MM-DD). Default: 365 days ago
            end_date: End date (YYYY-MM-DD). Default: today

        Returns:
            DataFrame with OHLCV data
        """
        coin_id = self._symbol_to_id(symbol)

        # Calculate days parameter
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
        else:
            days = 365  # Default to 1 year

        days = min(days, 365)  # CoinGecko free tier limit

        logger.info(f"Fetching {days} days of data for {coin_id}")

        # Fetch OHLC data (candles)
        ohlc_data = self._make_request(
            f"coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": days}
        )

        # Parse OHLC data
        df_ohlc = pd.DataFrame(
            ohlc_data,
            columns=["timestamp", "open", "high", "low", "close"]
        )
        df_ohlc["timestamp"] = pd.to_datetime(df_ohlc["timestamp"], unit="ms")
        df_ohlc.set_index("timestamp", inplace=True)

        # Fetch market chart for volume
        market_data = self._make_request(
            f"coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days}
        )

        # Parse volume data
        if "total_volumes" in market_data:
            df_vol = pd.DataFrame(
                market_data["total_volumes"],
                columns=["timestamp", "volume"]
            )
            df_vol["timestamp"] = pd.to_datetime(df_vol["timestamp"], unit="ms")
            df_vol.set_index("timestamp", inplace=True)

            # Resample volume to daily and merge
            df_vol_daily = df_vol.resample("D").sum()
            df_ohlc = df_ohlc.resample("D").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last"
            })
            df_ohlc = df_ohlc.join(df_vol_daily, how="left")
        else:
            df_ohlc["volume"] = 0

        # Clean up
        df_ohlc = df_ohlc.dropna(subset=["close"])
        df_ohlc = df_ohlc.sort_index()

        logger.info(f"Fetched {len(df_ohlc)} rows for {coin_id}")
        return df_ohlc

    def get_available_symbols(self) -> List[str]:
        """Get list of available cryptocurrencies."""
        data = self._make_request("coins/list")
        return [coin["id"] for coin in data]

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        coin_id = self._symbol_to_id(symbol)
        data = self._make_request(
            "simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"}
        )
        return data[coin_id]["usd"]

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market data for a symbol.

        Returns: Dict with price, market_cap, volume, price changes, etc.
        """
        coin_id = self._symbol_to_id(symbol)
        data = self._make_request(f"coins/{coin_id}")

        market_data = data.get("market_data", {})
        return {
            "current_price": market_data.get("current_price", {}).get("usd"),
            "market_cap": market_data.get("market_cap", {}).get("usd"),
            "total_volume": market_data.get("total_volume", {}).get("usd"),
            "price_change_24h": market_data.get("price_change_percentage_24h"),
            "price_change_7d": market_data.get("price_change_percentage_7d"),
            "price_change_30d": market_data.get("price_change_percentage_30d"),
            "ath": market_data.get("ath", {}).get("usd"),
            "ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd"),
            "circulating_supply": market_data.get("circulating_supply"),
        }
