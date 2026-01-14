#!/usr/bin/env python3
"""Run live multi-coin paper trading with Discord notifications."""
import sys
import os
import time
import schedule
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.coin_registry import get_coin_registry
from config.trading_config import TradingConfig
from config.model_config import ModelConfig
from trading.orchestrator import MultiCoinTradingOrchestrator
from utils.logger import setup_logger

logger = setup_logger("multi_coin_trading", log_dir=Path("logs"))


def run_daily_trading_cycle(orchestrator: MultiCoinTradingOrchestrator):
    """Execute a daily trading cycle."""
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY TRADING CYCLE - {datetime.now()}")
    logger.info(f"{'='*60}")

    try:
        results = orchestrator.run_trading_cycle()

        logger.info(f"\nCycle Results:")
        logger.info(f"  Status: {results['status']}")
        logger.info(f"  Signals Generated: {results.get('signals_generated', 0)}")
        logger.info(f"  Buys Executed: {results.get('buys_executed', 0)}")
        logger.info(f"  Sells Executed: {results.get('sells_executed', 0)}")
        logger.info(f"  Skipped: {results.get('skipped', 0)}")
        logger.info(f"  Portfolio Value: ${results.get('portfolio_value', 0):,.2f}")
        logger.info(f"  Total Return: {results.get('total_return_pct', 0):.2f}%")
        logger.info(f"  Open Positions: {results.get('num_positions', 0)}")

        # Send trade alerts for executed trades
        if orchestrator.notifier.enabled:
            execution = results.get("execution_details", {})

            # Alert for buys
            for buy in execution.get("buys", []):
                orchestrator.notifier.send_trade_alert(
                    action="BUY",
                    symbol=buy["coin"],
                    category=buy["category"],
                    quantity=buy["quantity"],
                    price=buy["price"],
                    confidence=buy["confidence"]
                )

            # Alert for sells
            for sell in execution.get("sells", []):
                orchestrator.notifier.send_trade_alert(
                    action="SELL",
                    symbol=sell["coin"],
                    category="",
                    quantity=sell["quantity"],
                    price=sell["price"],
                    confidence=0,
                    pnl=sell["pnl"],
                    pnl_pct=sell.get("pnl_pct", 0)
                )

        # Check if risk alert needed
        if results["status"] == "halted":
            orchestrator.send_risk_alert(
                "TRADING_HALTED",
                results.get("reason", "Unknown reason")
            )

    except Exception as e:
        logger.exception(f"Error in trading cycle: {e}")


def send_scheduled_report(orchestrator: MultiCoinTradingOrchestrator):
    """Send the 12-hour scheduled report."""
    logger.info("Sending scheduled 12-hour report...")
    orchestrator.send_trade_report(hours=12)


def main():
    """Main entry point for multi-coin paper trading."""
    logger.info("=" * 70)
    logger.info("MULTI-COIN CRYPTO ML TRADING SYSTEM")
    logger.info("=" * 70)

    # Show configuration
    registry = get_coin_registry()
    trading_config = TradingConfig()
    model_config = ModelConfig()

    logger.info(f"\nConfiguration:")
    logger.info(f"  Initial Capital: ${trading_config.initial_capital:,.2f}")
    logger.info(f"  Max Position: {trading_config.max_position_pct:.0%}")
    logger.info(f"  Max Exposure: {trading_config.max_total_exposure:.0%}")
    logger.info(f"  Max Drawdown: {trading_config.max_drawdown_pct:.0%}")
    logger.info(f"  Prediction Horizon: {model_config.prediction_horizon} days")

    # Check Discord webhook
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
    if discord_webhook:
        logger.info(f"  Discord Notifications: ENABLED")
    else:
        logger.info(f"  Discord Notifications: DISABLED (set DISCORD_WEBHOOK_URL)")

    # Show coin universe
    summary = registry.get_summary()
    logger.info(f"\nCoin Universe: {summary['tradeable_coins']} tradeable coins")
    for cat, info in summary["by_category"].items():
        if info["enabled"] and info["count"] > 0:
            logger.info(f"  {cat}: {info['count']} coins ({info['max_allocation']:.0%} max allocation)")

    # Initialize orchestrator
    logger.info("\nInitializing trading orchestrator...")
    orchestrator = MultiCoinTradingOrchestrator(
        trading_config=trading_config,
        model_config=model_config,
        db_path="data/trading.db",
        models_dir="models/saved",
        discord_webhook_url=discord_webhook
    )

    # Load models
    logger.info("Loading trained models...")
    orchestrator.load_models()

    if not orchestrator.models:
        logger.error("\nNo trained models found!")
        logger.error("Please run the following first:")
        logger.error("  1. python scripts/download_multi_coin_data.py")
        logger.error("  2. python scripts/train_multi_coin_models.py")
        return

    logger.info(f"Loaded models for {len(orchestrator.models)} coins")

    # Check for continuous mode vs single run
    continuous_mode = "--continuous" in sys.argv

    if continuous_mode:
        logger.info("\nRunning in CONTINUOUS mode")
        logger.info("  - Trading cycles: Daily at 09:00 and 21:00 UTC")
        logger.info("  - Reports to Discord: Every 12 hours")
        logger.info("Press Ctrl+C to stop")

        # Run first cycle immediately
        run_daily_trading_cycle(orchestrator)

        # Send initial report if Discord enabled
        if orchestrator.notifier.enabled:
            orchestrator.send_trade_report(hours=12)

        # Schedule trading cycles (twice daily for swing trading)
        schedule.every().day.at("09:00").do(
            run_daily_trading_cycle, orchestrator
        )
        schedule.every().day.at("21:00").do(
            run_daily_trading_cycle, orchestrator
        )

        # Schedule 12-hour reports
        schedule.every(12).hours.do(
            send_scheduled_report, orchestrator
        )

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    else:
        logger.info("\nRunning SINGLE trading cycle")
        logger.info("Use --continuous flag for automated trading with 12h reports")

        # Run single cycle
        run_daily_trading_cycle(orchestrator)

        # Send report if Discord enabled
        if orchestrator.notifier.enabled:
            logger.info("\nSending trade report to Discord...")
            orchestrator.send_trade_report(hours=12)

        logger.info("\nTrading cycle complete.")
        logger.info("For continuous trading: python scripts/run_multi_coin_trading.py --continuous")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTrading stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
