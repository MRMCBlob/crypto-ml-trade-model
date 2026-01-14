#!/usr/bin/env python3
"""Download historical cryptocurrency data."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetchers.coingecko_fetcher import CoinGeckoFetcher
from data.storage.database import TradingDatabase
from utils.logger import setup_logger

logger = setup_logger("download_data")


def main():
    """Download and store historical data for configured symbols."""
    # Initialize components
    fetcher = CoinGeckoFetcher()
    db = TradingDatabase("data/trading.db")

    # Symbols to download
    symbols = ["bitcoin", "ethereum", "solana", "cardano", "polkadot"]

    logger.info(f"Downloading data for {len(symbols)} symbols...")

    for symbol in symbols:
        try:
            logger.info(f"Fetching {symbol}...")
            df = fetcher.fetch_ohlcv(symbol)

            if len(df) > 0:
                db.save_ohlcv(df, symbol)
                logger.info(f"Saved {len(df)} rows for {symbol}")
            else:
                logger.warning(f"No data received for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue

    # Print summary
    counts = db.get_symbol_count()
    logger.info("Download complete. Records per symbol:")
    for symbol, count in counts.items():
        logger.info(f"  {symbol}: {count} records")


if __name__ == "__main__":
    main()
