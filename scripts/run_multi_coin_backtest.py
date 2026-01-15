#!/usr/bin/env python3
"""Run backtest across multiple coins."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from config.coin_registry import get_coin_registry
from config.trading_config import TradingConfig
from config.model_config import ModelConfig
from config.settings import get_settings
from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from trading.engine.paper_trader import PaperTradingEngine, OrderSide
from trading.engine.portfolio_manager import MultiCoinPortfolioManager
from trading.strategies.ml_swing_strategy import MLSwingStrategy, Signal
from trading.risk.risk_manager import RiskManager
from analysis.performance import PerformanceAnalyzer
from utils.logger import setup_logger

logger = setup_logger("multi_coin_backtest", log_dir=Path("logs"))


def load_models(models_dir: Path, registry) -> dict:
    """Load all available trained models."""
    models = {}
    strategies = {}

    for coin in registry.get_all_tradeable_coins():
        model_path = models_dir / coin.id
        if model_path.exists():
            try:
                model = HybridModel.load(str(model_path))
                models[coin.id] = model
                strategies[coin.id] = MLSwingStrategy(model, TradingConfig())
                logger.info(f"Loaded model for {coin.symbol}")
            except Exception as e:
                logger.warning(f"Failed to load model for {coin.symbol}: {e}")

    return models, strategies


def main():
    """Run multi-coin backtest."""
    logger.info("=" * 60)
    logger.info("MULTI-COIN BACKTESTING")
    logger.info("=" * 60)

    # Initialize
    settings = get_settings()
    registry = get_coin_registry()
    db = TradingDatabase(settings.database_path)
    trading_config = TradingConfig()
    model_config = ModelConfig()
    models_dir = Path("models/saved")

    # Load models
    logger.info("\nLoading trained models...")
    models, strategies = load_models(models_dir, registry)
    logger.info(f"Loaded {len(models)} models")

    if not models:
        logger.error("No models found. Run training first.")
        return

    # Initialize trading components
    engine = PaperTradingEngine(trading_config)
    portfolio_manager = MultiCoinPortfolioManager(engine, registry, trading_config)
    risk_manager = RiskManager(trading_config)

    # Prepare data for all coins
    logger.info("\nPreparing data...")
    coin_data = {}
    feature_cols = {}

    for coin_id in models.keys():
        df = db.load_ohlcv(coin_id)
        if len(df) >= model_config.sequence_length + 50:
            engineer = FeatureEngineer(df)
            df_features = engineer.add_all_features(model_config.prediction_horizon)
            df_clean = engineer.get_clean_features(dropna=True)

            if len(df_clean) > 0:
                coin_data[coin_id] = df_clean
                feature_cols[coin_id] = engineer.get_feature_columns()
                logger.info(f"  {coin_id}: {len(df_clean)} samples")

    if not coin_data:
        logger.error("No valid data for backtesting")
        return

    # Find common date range
    all_dates = set.intersection(*[set(df.index) for df in coin_data.values()])
    all_dates = sorted(list(all_dates))

    if len(all_dates) < 30:
        logger.error("Insufficient overlapping dates for backtest")
        return

    # Use last 90 days for backtest
    test_days = min(90, len(all_dates) - model_config.sequence_length)
    test_dates = all_dates[-test_days:]

    logger.info(f"\nBacktest period: {test_dates[0].date()} to {test_dates[-1].date()}")
    logger.info(f"Testing on {len(test_dates)} days across {len(coin_data)} coins")

    # Run backtest
    logger.info("\nStarting backtest...")
    risk_manager.update_daily_start(engine.get_portfolio_value())

    for i, date in enumerate(test_dates):
        if i % 10 == 0:
            logger.info(f"Processing day {i+1}/{len(test_dates)}: {date.date()}")

        # Get prices for this date
        prices = {}
        for coin_id, df in coin_data.items():
            if date in df.index:
                prices[coin_id] = df.loc[date, "close"]

        if not prices:
            continue

        # Update portfolio
        engine.update_prices(prices, date)

        # Check risk limits
        portfolio_value = engine.get_portfolio_value()
        is_allowed, reason = risk_manager.check_risk_limits(
            portfolio_value, len(engine.positions)
        )

        if not is_allowed and risk_manager.trading_halted:
            logger.warning(f"Trading halted on {date.date()}: {reason}")
            break

        # Generate signals for all coins
        signals = []
        for coin_id, df in coin_data.items():
            if date not in df.index or coin_id not in strategies:
                continue

            # Get data up to current date
            mask = df.index <= date
            current_data = df.loc[mask].tail(model_config.sequence_length + 50)

            if len(current_data) < model_config.sequence_length:
                continue

            position = engine.get_position(coin_id)
            position_dict = None
            if position:
                position_dict = {
                    "entry_price": position.entry_price,
                    "entry_timestamp": position.entry_timestamp,
                    "quantity": position.quantity
                }

            try:
                strategy = strategies[coin_id]
                signal = strategy.generate_signal(
                    current_data, feature_cols[coin_id], coin_id, position_dict
                )
                signals.append((coin_id, signal, prices[coin_id]))
            except Exception:
                continue

        # Execute SELL signals first
        for coin_id, signal, price in signals:
            if signal.signal == Signal.SELL:
                position = engine.get_position(coin_id)
                if position:
                    order = engine.submit_order(
                        coin_id, OrderSide.SELL, position.quantity, price, date
                    )
                    if order.status.value == "filled":
                        trade = engine.trades[-1]
                        risk_manager.record_trade_result(trade["pnl"])

        # Execute BUY signals
        buy_signals = [(c, s, p) for c, s, p in signals if s.signal == Signal.BUY]
        buy_signals.sort(key=lambda x: x[1].confidence, reverse=True)

        for coin_id, signal, price in buy_signals:
            if engine.get_position(coin_id):
                continue

            can_open, _ = portfolio_manager.can_open_position(coin_id, price * 0.1, prices)
            if not can_open:
                continue

            df = coin_data[coin_id]
            atr = df.loc[date, "atr_14"] if "atr_14" in df.columns else price * 0.02

            quantity = portfolio_manager.calculate_position_size(coin_id, price, atr, prices)
            if quantity > 0:
                engine.submit_order(coin_id, OrderSide.BUY, quantity, price, date)

        risk_manager.update_peak(portfolio_value)

    # Close all positions at end
    final_date = test_dates[-1]
    for coin_id, pos in list(engine.positions.items()):
        if coin_id in coin_data and final_date in coin_data[coin_id].index:
            price = coin_data[coin_id].loc[final_date, "close"]
            engine.submit_order(coin_id, OrderSide.SELL, pos.quantity, price, final_date)

    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    analyzer = PerformanceAnalyzer()

    if engine.snapshots:
        equity_data = [{"timestamp": s.timestamp, "value": s.total_value} for s in engine.snapshots]
        equity_df = pd.DataFrame(equity_data)
        equity_df.set_index("timestamp", inplace=True)
        equity_curve = equity_df["value"]

        report = analyzer.generate_report(equity_curve, engine.trades, trading_config.initial_capital)
        analyzer.print_report(report)

        # Category breakdown
        logger.info("\nTrades by category:")
        category_trades = {}
        for trade in engine.trades:
            coin = registry.get_coin(trade["symbol"])
            cat = coin.category.value if coin else "unknown"
            if cat not in category_trades:
                category_trades[cat] = {"count": 0, "pnl": 0}
            category_trades[cat]["count"] += 1
            category_trades[cat]["pnl"] += trade["pnl"]

        for cat, stats in sorted(category_trades.items()):
            logger.info(f"  {cat}: {stats['count']} trades, P&L: ${stats['pnl']:.2f}")

    else:
        logger.warning("No trading activity during backtest")

    # Final summary
    summary = engine.get_portfolio_summary()
    logger.info(f"\nFinal Portfolio: ${summary['total_value']:,.2f}")
    logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
