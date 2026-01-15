#!/usr/bin/env python3
"""Run live paper trading simulation."""
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from data.fetchers.coingecko_fetcher import CoinGeckoFetcher
from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from trading.engine.paper_trader import PaperTradingEngine, OrderSide
from trading.strategies.ml_swing_strategy import MLSwingStrategy, Signal
from trading.risk.risk_manager import RiskManager
from config.trading_config import TradingConfig
from config.model_config import ModelConfig
from utils.logger import setup_logger

logger = setup_logger("paper_trading", log_dir=Path("logs"))


def run_trading_cycle(
    fetcher: CoinGeckoFetcher,
    db: TradingDatabase,
    model: HybridModel,
    engine: PaperTradingEngine,
    strategy: MLSwingStrategy,
    risk_manager: RiskManager,
    symbol: str,
    feature_cols: list,
    model_config: ModelConfig
):
    """Run a single trading cycle."""
    timestamp = datetime.now()

    # Fetch latest data
    logger.info(f"Fetching latest data for {symbol}...")
    try:
        df = fetcher.fetch_ohlcv(symbol)
        if len(df) > 0:
            db.save_ohlcv(df, symbol)
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        # Use cached data
        df = db.load_ohlcv(symbol)

    if len(df) < model_config.sequence_length + 50:
        logger.error("Insufficient data")
        return

    # Feature engineering
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features(prediction_horizon=model_config.prediction_horizon)
    df_clean = engineer.get_clean_features(dropna=True)

    if len(df_clean) == 0:
        logger.error("No clean data after feature engineering")
        return

    current_price = df_clean["close"].iloc[-1]
    logger.info(f"Current {symbol} price: ${current_price:,.2f}")

    # Update portfolio with current prices
    engine.update_prices({symbol: current_price}, timestamp)

    # Check risk limits
    portfolio_value = engine.get_portfolio_value()
    is_allowed, reason = risk_manager.check_risk_limits(
        portfolio_value, len(engine.positions)
    )

    if not is_allowed:
        logger.warning(f"Trading restricted: {reason}")
        return

    # Get current position
    position = engine.get_position(symbol)
    position_dict = None
    if position:
        position_dict = {
            "entry_price": position.entry_price,
            "entry_timestamp": position.entry_timestamp,
            "quantity": position.quantity
        }
        logger.info(
            f"Current position: {position.quantity:.6f} {symbol} "
            f"@ ${position.entry_price:.2f} "
            f"(P&L: ${position.unrealized_pnl:.2f})"
        )

    # Generate trading signal
    try:
        signal = strategy.generate_signal(df_clean, feature_cols, symbol, position_dict)
        logger.info(
            f"Signal: {signal.signal.name} | "
            f"Confidence: {signal.confidence:.2%} | "
            f"Reason: {signal.reasoning}"
        )
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return

    # Execute trades
    if signal.signal == Signal.BUY and position is None:
        atr = df_clean["atr_14"].iloc[-1] if "atr_14" in df_clean.columns else current_price * 0.02
        quantity = strategy.calculate_position_size(
            portfolio_value, current_price, atr, engine.get_exposure()
        )

        if quantity > 0:
            logger.info(f"Executing BUY: {quantity:.6f} {symbol} @ ${current_price:.2f}")
            order = engine.submit_order(symbol, OrderSide.BUY, quantity, current_price, timestamp)
            logger.info(f"Order status: {order.status.value}")

    elif signal.signal == Signal.SELL and position is not None:
        logger.info(f"Executing SELL: {position.quantity:.6f} {symbol} @ ${current_price:.2f}")
        order = engine.submit_order(symbol, OrderSide.SELL, position.quantity, current_price, timestamp)
        logger.info(f"Order status: {order.status.value}")

        if order.status.value == "filled":
            trade = engine.trades[-1]
            risk_manager.record_trade_result(trade["pnl"])
            logger.info(f"Trade P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")

    # Update risk manager
    risk_manager.update_peak(portfolio_value)

    # Print portfolio summary
    summary = engine.get_portfolio_summary()
    logger.info(f"\n--- Portfolio Summary ---")
    logger.info(f"Total Value:  ${summary['total_value']:,.2f}")
    logger.info(f"Cash:         ${summary['cash_balance']:,.2f}")
    logger.info(f"Positions:    ${summary['positions_value']:,.2f}")
    logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
    logger.info(f"Trades:       {summary['total_trades']} (Win rate: {summary['win_rate']*100:.1f}%)")


def main():
    """Run paper trading with periodic updates."""
    logger.info("=" * 60)
    logger.info("CRYPTO ML PAPER TRADING")
    logger.info("=" * 60)

    # Initialize components
    settings = get_settings()
    fetcher = CoinGeckoFetcher()
    db = TradingDatabase(settings.database_path)
    trading_config = TradingConfig()
    model_config = ModelConfig()

    # Load model
    model_path = Path("models/saved/hybrid_model")
    if not model_path.exists():
        logger.error("Model not found. Run training first:")
        logger.error("  python scripts/download_data.py")
        logger.error("  python scripts/train_model.py")
        return

    logger.info("Loading trained model...")
    model = HybridModel.load(str(model_path))

    # Initialize trading components
    engine = PaperTradingEngine(trading_config)
    strategy = MLSwingStrategy(model, trading_config)
    risk_manager = RiskManager(trading_config)

    # Get feature columns from a sample
    symbol = "bitcoin"
    df_sample = db.load_ohlcv(symbol)
    if len(df_sample) == 0:
        logger.info("Downloading initial data...")
        df_sample = fetcher.fetch_ohlcv(symbol)
        db.save_ohlcv(df_sample, symbol)

    engineer = FeatureEngineer(df_sample)
    engineer.add_all_features()
    feature_cols = engineer.get_feature_columns()

    logger.info(f"\nStarting paper trading for {symbol}")
    logger.info(f"Initial capital: ${trading_config.initial_capital:,.2f}")
    logger.info(f"Update interval: Daily (swing trading)")
    logger.info("Press Ctrl+C to stop\n")

    # Set daily start for risk management
    risk_manager.update_daily_start(engine.get_portfolio_value())

    # Run initial cycle
    run_trading_cycle(
        fetcher, db, model, engine, strategy, risk_manager,
        symbol, feature_cols, model_config
    )

    # For swing trading, we typically update once per day
    # In a real scenario, this would be scheduled via cron or similar
    logger.info("\nPaper trading cycle complete.")
    logger.info("For continuous trading, schedule this script to run daily.")
    logger.info("Example cron entry: 0 9 * * * python scripts/run_paper_trading.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nPaper trading stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
