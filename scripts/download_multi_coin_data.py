#!/usr/bin/env python3
"""Download historical data for all registered coins."""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.coin_registry import get_coin_registry, CoinCategory
from config.settings import get_settings
from data.fetchers.coingecko_fetcher import CoinGeckoFetcher
from data.storage.database import TradingDatabase
from utils.logger import setup_logger

logger = setup_logger("download_multi_coin", log_dir=Path("logs"))


def main():
    """Download data for all tradeable coins across categories."""
    logger.info("=" * 60)
    logger.info("MULTI-COIN DATA DOWNLOADER")
    logger.info("=" * 60)

    # Initialize
    settings = get_settings()
    registry = get_coin_registry()
    fetcher = CoinGeckoFetcher(api_key=settings.coingecko_api_key or None)
    db = TradingDatabase(settings.database_path)

    if settings.coingecko_api_key:
        logger.info("Using CoinGecko API key for higher rate limits")
    else:
        logger.warning("No API key - using free tier (slower, may hit rate limits)")

    # Get all tradeable coins
    coins = registry.get_all_tradeable_coins()
    logger.info(f"Found {len(coins)} tradeable coins")

    # Print summary by category
    summary = registry.get_summary()
    logger.info("\nCoins by category:")
    for cat, info in summary["by_category"].items():
        if info["enabled"]:
            logger.info(f"  {cat}: {info['count']} coins - {', '.join(info['coins'])}")

    # Download data for each coin
    successful = 0
    failed = []

    for i, coin in enumerate(coins, 1):
        logger.info(f"\n[{i}/{len(coins)}] Fetching {coin.name} ({coin.symbol})...")

        try:
            df = fetcher.fetch_ohlcv(coin.id)

            if len(df) > 0:
                db.save_ohlcv(df, coin.id)
                logger.info(f"  Saved {len(df)} rows for {coin.symbol}")
                successful += 1
            else:
                logger.warning(f"  No data received for {coin.symbol}")
                failed.append(coin.symbol)

            # Rate limiting - CoinGecko free tier is ~10 calls/min
            # With API key it's ~30 calls/min
            delay = 3 if settings.coingecko_api_key else 8
            time.sleep(delay)

        except Exception as e:
            logger.error(f"  Error fetching {coin.symbol}: {e}")
            failed.append(coin.symbol)
            time.sleep(60)  # Long delay on error (likely rate limited)
            continue

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}/{len(coins)}")

    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")

    # Print database summary
    counts = db.get_symbol_count()
    logger.info("\nRecords per coin:")
    for symbol, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        coin = registry.get_coin(symbol)
        cat = f"[{coin.category.value}]" if coin else ""
        logger.info(f"  {symbol}: {count} rows {cat}")


if __name__ == "__main__":
    main()
