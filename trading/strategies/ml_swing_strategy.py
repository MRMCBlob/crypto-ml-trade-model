"""ML-based swing trading strategy."""
import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from models.hybrid_model import HybridModel
from config.trading_config import TradingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Signal(Enum):
    """Trading signal enum."""
    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class TradeSignal:
    """Trading signal with metadata."""
    signal: Signal
    confidence: float
    symbol: str
    timestamp: pd.Timestamp
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None


class MLSwingStrategy:
    """
    ML-based swing trading strategy.

    Converts model predictions into actionable trading signals
    with risk management parameters.
    """

    def __init__(
        self,
        model: HybridModel,
        config: Optional[TradingConfig] = None
    ):
        """
        Initialize strategy.

        Args:
            model: Trained hybrid ML model
            config: Trading configuration
        """
        self.model = model
        self.config = config or TradingConfig()

    def generate_signal(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        symbol: str,
        current_position: Optional[Dict[str, Any]] = None
    ) -> TradeSignal:
        """
        Generate trading signal from model predictions.

        Args:
            df: DataFrame with price data and features
            feature_cols: Feature column names for model
            symbol: Trading symbol
            current_position: Existing position if any

        Returns:
            TradeSignal with entry/exit parameters
        """
        # Get model predictions
        predictions = self.model.predict(df, feature_cols)
        latest_prob = predictions["ensemble_prob"][-1]

        # Get current price and ATR for risk parameters
        latest_price = df["close"].iloc[-1]
        latest_atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else latest_price * 0.02
        timestamp = df.index[-1]

        # Determine signal based on thresholds
        signal = Signal.HOLD
        reasoning = None
        stop_loss = None
        take_profit = None

        if latest_prob >= self.config.buy_threshold and latest_prob >= self.config.min_confidence:
            if current_position is None:
                # No position - generate BUY signal
                signal = Signal.BUY
                stop_loss = latest_price - (self.config.stop_loss_atr_multiple * latest_atr)
                take_profit = latest_price + (self.config.take_profit_atr_multiple * latest_atr)
                reasoning = f"High probability ({latest_prob:.1%}) above buy threshold"
            else:
                # Already have position - HOLD
                signal = Signal.HOLD
                reasoning = "Already in position, maintaining"

        elif latest_prob <= self.config.sell_threshold:
            if current_position is not None:
                # Have position - generate SELL signal
                signal = Signal.SELL
                reasoning = f"Low probability ({latest_prob:.1%}) below sell threshold"
            else:
                # No position to sell - HOLD (could implement short selling here)
                signal = Signal.HOLD
                reasoning = "Bearish signal but no position to close"

        else:
            # Probability in neutral zone
            signal = Signal.HOLD
            reasoning = f"Neutral probability ({latest_prob:.1%}), no action"

        # Check position timeout (force exit after max holding period)
        if current_position and signal == Signal.HOLD:
            entry_time = current_position.get("entry_timestamp")
            if entry_time:
                days_held = (timestamp - entry_time).days
                if days_held >= self.config.max_holding_days:
                    signal = Signal.SELL
                    reasoning = f"Max holding period ({days_held} days) reached"

        return TradeSignal(
            signal=signal,
            confidence=latest_prob,
            symbol=symbol,
            timestamp=timestamp,
            current_price=latest_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )

    def check_stop_loss_take_profit(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[Signal]:
        """
        Check if stop-loss or take-profit has been hit.

        Args:
            position: Current position dict with entry_price, stop_loss, take_profit
            current_price: Current market price

        Returns:
            SELL signal if stop/target hit, None otherwise
        """
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")

        if stop_loss and current_price <= stop_loss:
            logger.info(f"Stop-loss hit: {current_price:.2f} <= {stop_loss:.2f}")
            return Signal.SELL

        if take_profit and current_price >= take_profit:
            logger.info(f"Take-profit hit: {current_price:.2f} >= {take_profit:.2f}")
            return Signal.SELL

        return None

    def calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        atr: float,
        current_exposure: float = 0.0
    ) -> float:
        """
        Calculate position size using volatility-adjusted sizing.

        Args:
            portfolio_value: Total portfolio value
            current_price: Current asset price
            atr: Average True Range
            current_exposure: Current portfolio exposure

        Returns:
            Position size in units (quantity to buy)
        """
        # Check exposure limits
        available_exposure = self.config.max_total_exposure - current_exposure
        if available_exposure <= 0:
            logger.warning("Max exposure reached, no new positions allowed")
            return 0.0

        # Risk-based position sizing
        risk_amount = portfolio_value * self.config.risk_per_trade_pct
        stop_distance = atr * self.config.stop_loss_atr_multiple

        if stop_distance <= 0:
            return 0.0

        # Position size based on risk
        risk_based_quantity = risk_amount / stop_distance

        # Position size based on max position limit
        max_position_value = portfolio_value * min(
            self.config.max_position_pct,
            available_exposure
        )
        max_based_quantity = max_position_value / current_price

        # Use the smaller of the two
        quantity = min(risk_based_quantity, max_based_quantity)

        logger.debug(
            f"Position sizing: risk-based={risk_based_quantity:.6f}, "
            f"max-based={max_based_quantity:.6f}, final={quantity:.6f}"
        )

        return quantity
