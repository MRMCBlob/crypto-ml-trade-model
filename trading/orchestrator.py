"""Multi-coin trading orchestrator."""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.coin_registry import (
    CoinRegistry, CoinInfo, CoinCategory,
    get_coin_registry
)
from config.trading_config import TradingConfig
from config.model_config import ModelConfig
from data.fetchers.coingecko_fetcher import CoinGeckoFetcher
from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from trading.engine.paper_trader import PaperTradingEngine, OrderSide
from trading.engine.portfolio_manager import MultiCoinPortfolioManager
from trading.strategies.ml_swing_strategy import MLSwingStrategy, Signal
from trading.risk.risk_manager import RiskManager
from utils.logger import get_logger
from utils.notifications import DiscordNotifier, ReportGenerator

logger = get_logger(__name__)


@dataclass
class CoinSignal:
    """Trading signal for a specific coin."""
    coin_id: str
    symbol: str
    category: CoinCategory
    signal: Signal
    confidence: float
    current_price: float
    reasoning: str
    timestamp: datetime


class MultiCoinTradingOrchestrator:
    """
    Orchestrates trading across multiple coins with different categories.

    Coordinates data fetching, model inference, signal generation,
    and trade execution across a diversified crypto portfolio.
    """

    def __init__(
        self,
        trading_config: Optional[TradingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        db_path: str = "storage/trading.db",
        models_dir: str = "models/saved",
        discord_webhook_url: Optional[str] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            trading_config: Trading configuration
            model_config: Model configuration
            db_path: Path to SQLite database
            models_dir: Directory containing trained models
            discord_webhook_url: Discord webhook URL for notifications
        """
        self.trading_config = trading_config or TradingConfig()
        self.model_config = model_config or ModelConfig()
        self.models_dir = Path(models_dir)

        # Initialize components
        self.registry = get_coin_registry()
        self.fetcher = CoinGeckoFetcher()
        self.db = TradingDatabase(db_path)
        self.engine = PaperTradingEngine(self.trading_config)
        self.portfolio_manager = MultiCoinPortfolioManager(
            self.engine, self.registry, self.trading_config
        )
        self.risk_manager = RiskManager(self.trading_config)

        # Models and strategies per coin
        self.models: Dict[str, HybridModel] = {}
        self.strategies: Dict[str, MLSwingStrategy] = {}
        self.feature_cols: Dict[str, List[str]] = {}

        # Current prices cache
        self.current_prices: Dict[str, float] = {}

        # Discord notifications
        webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self.notifier = DiscordNotifier(webhook_url)
        self.report_generator = ReportGenerator(
            self.engine,
            self.portfolio_manager,
            self.risk_manager,
            self.registry
        )

        # Track last report time
        self.last_report_time: Optional[datetime] = None

        logger.info("Multi-coin trading orchestrator initialized")
        if self.notifier.enabled:
            logger.info("Discord notifications enabled")

    def load_models(self, coin_ids: Optional[List[str]] = None):
        """
        Load trained models for specified coins.

        Args:
            coin_ids: List of coin IDs to load models for.
                     If None, loads all available models.
        """
        if coin_ids is None:
            coin_ids = self.registry.get_coin_ids()

        loaded = 0
        for coin_id in coin_ids:
            model_path = self.models_dir / coin_id
            if model_path.exists():
                try:
                    model = HybridModel.load(str(model_path))
                    self.models[coin_id] = model
                    self.strategies[coin_id] = MLSwingStrategy(model, self.trading_config)
                    loaded += 1
                    logger.info(f"Loaded model for {coin_id}")
                except Exception as e:
                    logger.error(f"Failed to load model for {coin_id}: {e}")
            else:
                logger.debug(f"No model found for {coin_id}")

        logger.info(f"Loaded {loaded} models out of {len(coin_ids)} coins")

    def fetch_current_prices(self, coin_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Fetch current prices for all coins.

        Args:
            coin_ids: Specific coins to fetch. If None, fetches all tradeable.

        Returns:
            Dictionary of coin_id -> current price
        """
        if coin_ids is None:
            coin_ids = list(self.models.keys())

        prices = {}
        for coin_id in coin_ids:
            try:
                price = self.fetcher.get_current_price(coin_id)
                prices[coin_id] = price
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch price for {coin_id}: {e}")
                # Try to get last known price from DB
                df = self.db.load_ohlcv(coin_id)
                if len(df) > 0:
                    prices[coin_id] = df["close"].iloc[-1]

        self.current_prices = prices
        return prices

    def update_coin_data(self, coin_id: str) -> bool:
        """
        Update data for a single coin.

        Args:
            coin_id: Coin to update

        Returns:
            True if successful
        """
        try:
            df = self.fetcher.fetch_ohlcv(coin_id)
            if len(df) > 0:
                self.db.save_ohlcv(df, coin_id)
                return True
        except Exception as e:
            logger.error(f"Failed to update data for {coin_id}: {e}")
        return False

    def generate_signal_for_coin(self, coin_id: str) -> Optional[CoinSignal]:
        """
        Generate trading signal for a single coin.

        Args:
            coin_id: Coin to analyze

        Returns:
            CoinSignal or None if unable to generate
        """
        if coin_id not in self.models:
            return None

        coin_info = self.registry.get_coin(coin_id)
        if not coin_info or not coin_info.enabled:
            return None

        try:
            # Load and process data
            df = self.db.load_ohlcv(coin_id)
            if len(df) < self.model_config.sequence_length + 50:
                logger.warning(f"Insufficient data for {coin_id}")
                return None

            # Feature engineering
            engineer = FeatureEngineer(df)
            df_features = engineer.add_all_features(
                prediction_horizon=self.model_config.prediction_horizon
            )
            feature_cols = engineer.get_feature_columns()
            self.feature_cols[coin_id] = feature_cols

            df_clean = engineer.get_clean_features(dropna=True)
            if len(df_clean) == 0:
                return None

            current_price = df_clean["close"].iloc[-1]

            # Get current position
            position = self.engine.get_position(coin_id)
            position_dict = None
            if position:
                position_dict = {
                    "entry_price": position.entry_price,
                    "entry_timestamp": position.entry_timestamp,
                    "quantity": position.quantity
                }

            # Generate signal
            strategy = self.strategies[coin_id]
            signal = strategy.generate_signal(
                df_clean, feature_cols, coin_id, position_dict
            )

            return CoinSignal(
                coin_id=coin_id,
                symbol=coin_info.symbol,
                category=coin_info.category,
                signal=signal.signal,
                confidence=signal.confidence,
                current_price=current_price,
                reasoning=signal.reasoning or "",
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error generating signal for {coin_id}: {e}")
            return None

    def generate_all_signals(self) -> List[CoinSignal]:
        """
        Generate signals for all coins with trained models.

        Returns:
            List of CoinSignals
        """
        signals = []

        for coin_id in self.models.keys():
            signal = self.generate_signal_for_coin(coin_id)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Signal for {signal.symbol}: {signal.signal.name} "
                    f"({signal.confidence:.1%}) - {signal.reasoning}"
                )

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals

    def execute_signals(self, signals: List[CoinSignal]) -> Dict[str, Any]:
        """
        Execute trading signals with portfolio management.

        Args:
            signals: List of signals to potentially execute

        Returns:
            Execution summary
        """
        executed_buys = []
        executed_sells = []
        skipped = []

        timestamp = datetime.now()

        # First process SELL signals (free up capital)
        for signal in signals:
            if signal.signal != Signal.SELL:
                continue

            position = self.engine.get_position(signal.coin_id)
            if position is None:
                continue

            # Execute sell
            order = self.engine.submit_order(
                signal.coin_id,
                OrderSide.SELL,
                position.quantity,
                signal.current_price,
                timestamp
            )

            if order.status.value == "filled":
                trade = self.engine.trades[-1]
                self.risk_manager.record_trade_result(trade["pnl"])
                executed_sells.append({
                    "coin": signal.symbol,
                    "quantity": position.quantity,
                    "price": signal.current_price,
                    "pnl": trade["pnl"]
                })
                logger.info(
                    f"SOLD {signal.symbol}: {position.quantity:.6f} @ "
                    f"${signal.current_price:.2f}, P&L: ${trade['pnl']:.2f}"
                )

        # Check risk limits before buying
        portfolio_value = self.engine.get_portfolio_value()
        is_allowed, reason = self.risk_manager.check_risk_limits(
            portfolio_value, len(self.engine.positions)
        )

        if not is_allowed:
            logger.warning(f"Trading restricted: {reason}")
            return {
                "sells": executed_sells,
                "buys": [],
                "skipped": [s.symbol for s in signals if s.signal == Signal.BUY],
                "reason": reason
            }

        # Process BUY signals (sorted by confidence)
        buy_signals = [s for s in signals if s.signal == Signal.BUY]

        for signal in buy_signals:
            # Check if we already have a position
            if self.engine.get_position(signal.coin_id):
                skipped.append({"coin": signal.symbol, "reason": "Already in position"})
                continue

            # Check category limits
            can_open, reason = self.portfolio_manager.can_open_position(
                signal.coin_id,
                self.engine.get_portfolio_value() * 0.1,  # Estimate
                self.current_prices
            )

            if not can_open:
                skipped.append({"coin": signal.symbol, "reason": reason})
                continue

            # Calculate position size
            df = self.db.load_ohlcv(signal.coin_id)
            engineer = FeatureEngineer(df)
            engineer.add_all_features()
            df_clean = engineer.get_clean_features(dropna=True)

            atr = df_clean["atr_14"].iloc[-1] if "atr_14" in df_clean.columns else signal.current_price * 0.02

            quantity = self.portfolio_manager.calculate_position_size(
                signal.coin_id,
                signal.current_price,
                atr,
                self.current_prices
            )

            if quantity <= 0:
                skipped.append({"coin": signal.symbol, "reason": "Position size too small"})
                continue

            # Execute buy
            order = self.engine.submit_order(
                signal.coin_id,
                OrderSide.BUY,
                quantity,
                signal.current_price,
                timestamp
            )

            if order.status.value == "filled":
                executed_buys.append({
                    "coin": signal.symbol,
                    "category": signal.category.value,
                    "quantity": quantity,
                    "price": signal.current_price,
                    "confidence": signal.confidence
                })
                logger.info(
                    f"BOUGHT {signal.symbol}: {quantity:.6f} @ "
                    f"${signal.current_price:.2f} [{signal.category.value}]"
                )
            else:
                skipped.append({
                    "coin": signal.symbol,
                    "reason": order.rejection_reason or "Order rejected"
                })

        # Update risk manager
        self.risk_manager.update_peak(self.engine.get_portfolio_value())

        return {
            "sells": executed_sells,
            "buys": executed_buys,
            "skipped": skipped
        }

    def run_trading_cycle(self) -> Dict[str, Any]:
        """
        Run a complete trading cycle.

        Returns:
            Cycle results summary
        """
        logger.info("=" * 60)
        logger.info("STARTING TRADING CYCLE")
        logger.info("=" * 60)

        # 1. Fetch current prices
        logger.info("Fetching current prices...")
        self.fetch_current_prices()

        # 2. Update portfolio state
        self.engine.update_prices(self.current_prices, datetime.now())

        # 3. Check risk limits
        portfolio_value = self.engine.get_portfolio_value()
        self.risk_manager.update_daily_start(portfolio_value)

        is_allowed, reason = self.risk_manager.check_risk_limits(
            portfolio_value, len(self.engine.positions)
        )

        if not is_allowed and self.risk_manager.trading_halted:
            logger.error(f"Trading halted: {reason}")
            return {"status": "halted", "reason": reason}

        # 4. Generate signals
        logger.info("Generating signals for all coins...")
        signals = self.generate_all_signals()
        logger.info(f"Generated {len(signals)} signals")

        # 5. Execute signals
        logger.info("Executing signals...")
        execution_results = self.execute_signals(signals)

        # 6. Print portfolio summary
        self.portfolio_manager.print_portfolio_summary(self.current_prices)

        # 7. Get portfolio summary
        summary = self.engine.get_portfolio_summary()

        return {
            "status": "completed",
            "signals_generated": len(signals),
            "buys_executed": len(execution_results["buys"]),
            "sells_executed": len(execution_results["sells"]),
            "skipped": len(execution_results["skipped"]),
            "portfolio_value": summary["total_value"],
            "total_return_pct": summary["total_return_pct"],
            "num_positions": summary["num_positions"],
            "execution_details": execution_results
        }

    def print_signals_summary(self, signals: List[CoinSignal]):
        """Print formatted signals summary."""
        print("\n" + "=" * 70)
        print("TRADING SIGNALS SUMMARY")
        print("=" * 70)

        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        hold_signals = [s for s in signals if s.signal == Signal.HOLD]

        if buy_signals:
            print("\n--- BUY SIGNALS ---")
            for s in sorted(buy_signals, key=lambda x: x.confidence, reverse=True):
                print(
                    f"  {s.symbol:8} | Confidence: {s.confidence:5.1%} | "
                    f"Price: ${s.current_price:>10,.2f} | "
                    f"[{s.category.value}] | {s.reasoning}"
                )

        if sell_signals:
            print("\n--- SELL SIGNALS ---")
            for s in sell_signals:
                print(
                    f"  {s.symbol:8} | Confidence: {s.confidence:5.1%} | "
                    f"Price: ${s.current_price:>10,.2f} | "
                    f"[{s.category.value}] | {s.reasoning}"
                )

        if hold_signals:
            print(f"\n--- HOLD ({len(hold_signals)} coins) ---")

        print("=" * 70 + "\n")

    def send_trade_report(self, hours: int = 12) -> bool:
        """
        Send trading report to Discord.

        Args:
            hours: Report period in hours

        Returns:
            True if sent successfully
        """
        if not self.notifier.enabled:
            logger.warning("Discord notifications not configured")
            return False

        try:
            # Ensure we have current prices
            if not self.current_prices:
                self.fetch_current_prices()

            # Generate report
            report = self.report_generator.generate_report(
                self.current_prices,
                hours=hours
            )

            # Send to Discord
            success = self.notifier.send_trade_report(report)

            if success:
                self.last_report_time = datetime.now()
                logger.info(f"Trade report sent successfully ({hours}h period)")
            else:
                logger.error("Failed to send trade report")

            return success

        except Exception as e:
            logger.error(f"Error sending trade report: {e}")
            return False

    def send_trade_alert(
        self,
        action: str,
        signal: "CoinSignal",
        quantity: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None
    ) -> bool:
        """
        Send instant trade alert to Discord.

        Args:
            action: BUY or SELL
            signal: The coin signal
            quantity: Trade quantity
            pnl: Realized P&L (for sells)
            pnl_pct: Realized P&L percentage (for sells)

        Returns:
            True if sent successfully
        """
        if not self.notifier.enabled:
            return False

        return self.notifier.send_trade_alert(
            action=action,
            symbol=signal.symbol,
            category=signal.category.value,
            quantity=quantity,
            price=signal.current_price,
            confidence=signal.confidence,
            pnl=pnl,
            pnl_pct=pnl_pct
        )

    def send_risk_alert(self, alert_type: str, message: str) -> bool:
        """
        Send risk management alert to Discord.

        Args:
            alert_type: Type of alert
            message: Alert message

        Returns:
            True if sent successfully
        """
        if not self.notifier.enabled:
            return False

        details = self.risk_manager.get_status(
            self.engine.get_portfolio_value(),
            len(self.engine.positions)
        )

        return self.notifier.send_risk_alert(alert_type, message, details)

    def check_and_send_scheduled_report(self, interval_hours: int = 12) -> bool:
        """
        Check if it's time to send a scheduled report and send if due.

        Args:
            interval_hours: Hours between reports

        Returns:
            True if report was sent
        """
        now = datetime.now()

        # If never sent, send now
        if self.last_report_time is None:
            return self.send_trade_report(hours=interval_hours)

        # Check if interval has passed
        hours_since_last = (now - self.last_report_time).total_seconds() / 3600

        if hours_since_last >= interval_hours:
            return self.send_trade_report(hours=interval_hours)

        return False
