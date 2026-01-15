#!/usr/bin/env python3
"""Run backtest on historical data."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.settings import get_settings
from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from trading.engine.paper_trader import PaperTradingEngine, OrderSide
from trading.strategies.ml_swing_strategy import MLSwingStrategy, Signal
from trading.risk.risk_manager import RiskManager
from analysis.performance import PerformanceAnalyzer
from config.trading_config import TradingConfig
from config.model_config import ModelConfig
from utils.logger import setup_logger

logger = setup_logger("backtest")


def main():
    """Run walk-forward backtest."""
    # Initialize
    settings = get_settings()
    db = TradingDatabase(settings.database_path)
    trading_config = TradingConfig()
    model_config = ModelConfig()

    # Load model
    model_path = Path("models/saved/hybrid_model")
    if not model_path.exists():
        logger.error("Model not found. Run 'python scripts/train_model.py' first.")
        return

    logger.info("Loading trained model...")
    model = HybridModel.load(str(model_path))

    # Load data
    symbol = "bitcoin"
    logger.info(f"Loading data for {symbol}...")
    df = db.load_ohlcv(symbol)

    if len(df) < 100:
        logger.error("Insufficient data for backtesting")
        return

    # Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features(prediction_horizon=model_config.prediction_horizon)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.get_clean_features(dropna=True)

    # Split into train/test (use last 90 days for test)
    test_days = 90
    split_idx = len(df_clean) - test_days
    df_test = df_clean.iloc[split_idx:]

    logger.info(f"Backtesting on {len(df_test)} days of data")
    logger.info(f"Period: {df_test.index[0].date()} to {df_test.index[-1].date()}")

    # Initialize trading components
    engine = PaperTradingEngine(trading_config)
    strategy = MLSwingStrategy(model, trading_config)
    risk_manager = RiskManager(trading_config)

    # Run backtest
    logger.info("\nStarting backtest...")
    current_position = None

    for i in range(model_config.sequence_length, len(df_test)):
        # Get data up to current point (no lookahead)
        current_data = df_test.iloc[:i+1]
        timestamp = current_data.index[-1]
        current_price = current_data["close"].iloc[-1]

        # Update prices
        engine.update_prices({symbol: current_price}, timestamp)

        # Check risk limits
        portfolio_value = engine.get_portfolio_value()
        is_allowed, reason = risk_manager.check_risk_limits(
            portfolio_value, len(engine.positions)
        )

        if not is_allowed:
            if risk_manager.trading_halted:
                logger.warning(f"Trading halted: {reason}")
                break
            continue

        # Get current position
        position = engine.get_position(symbol)
        position_dict = None
        if position:
            position_dict = {
                "entry_price": position.entry_price,
                "entry_timestamp": position.entry_timestamp,
                "quantity": position.quantity
            }

        # Generate signal
        try:
            signal = strategy.generate_signal(
                current_data, feature_cols, symbol, position_dict
            )
        except Exception as e:
            continue

        # Execute trades based on signal
        if signal.signal == Signal.BUY and position is None:
            # Calculate position size
            atr = current_data["atr_14"].iloc[-1] if "atr_14" in current_data.columns else current_price * 0.02
            quantity = strategy.calculate_position_size(
                portfolio_value, current_price, atr, engine.get_exposure()
            )

            if quantity > 0:
                order = engine.submit_order(
                    symbol, OrderSide.BUY, quantity, current_price, timestamp
                )

        elif signal.signal == Signal.SELL and position is not None:
            order = engine.submit_order(
                symbol, OrderSide.SELL, position.quantity, current_price, timestamp
            )
            if order.status.value == "filled":
                # Record trade result for risk manager
                trade = engine.trades[-1]
                risk_manager.record_trade_result(trade["pnl"])

        # Update risk manager
        risk_manager.update_peak(portfolio_value)

    # Close any remaining positions at end of backtest
    final_timestamp = df_test.index[-1]
    final_price = df_test["close"].iloc[-1]
    for sym, pos in list(engine.positions.items()):
        engine.submit_order(sym, OrderSide.SELL, pos.quantity, final_price, final_timestamp)

    # Generate performance report
    logger.info("\nGenerating performance report...")
    analyzer = PerformanceAnalyzer()

    # Create equity curve from snapshots
    if engine.snapshots:
        equity_data = [
            {"timestamp": s.timestamp, "value": s.total_value}
            for s in engine.snapshots
        ]
        equity_df = pd.DataFrame(equity_data)
        equity_df.set_index("timestamp", inplace=True)
        equity_curve = equity_df["value"]

        report = analyzer.generate_report(
            equity_curve, engine.trades, trading_config.initial_capital
        )
        analyzer.print_report(report)

        # Compare to buy-and-hold
        bh_return = (df_test["close"].iloc[-1] / df_test["close"].iloc[0] - 1) * 100
        logger.info(f"\nBuy & Hold Return: {bh_return:.2f}%")
        logger.info(f"Strategy Return:   {report.total_return_pct:.2f}%")
        logger.info(f"Outperformance:    {report.total_return_pct - bh_return:.2f}%")
    else:
        logger.warning("No trading activity during backtest period")

    # Print portfolio summary
    summary = engine.get_portfolio_summary()
    logger.info(f"\nFinal Portfolio: ${summary['total_value']:,.2f}")
    logger.info(f"Total Trades: {summary['total_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
