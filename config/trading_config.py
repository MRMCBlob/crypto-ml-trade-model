"""Trading-specific configuration."""
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """Configuration for trading strategy and risk management."""

    # Portfolio settings
    initial_capital: float = 10000.0

    # Position sizing
    max_position_pct: float = 0.20      # Max 20% of portfolio per position
    max_total_exposure: float = 0.80     # Max 80% total exposure
    risk_per_trade_pct: float = 0.02     # Risk 2% per trade

    # Risk limits
    max_daily_loss_pct: float = 0.05     # 5% max daily loss
    max_drawdown_pct: float = 0.15       # 15% max drawdown before halt
    max_consecutive_losses: int = 5       # Pause after 5 consecutive losses
    max_positions: int = 5                # Max concurrent positions

    # Stop-loss and take-profit (ATR multiples)
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0

    # Trading costs simulation
    trading_fee_pct: float = 0.001       # 0.1% per trade
    slippage_pct: float = 0.0005         # 0.05% slippage

    # Signal thresholds
    buy_threshold: float = 0.60          # Probability threshold to buy
    sell_threshold: float = 0.40         # Probability threshold to sell
    min_confidence: float = 0.55         # Minimum confidence to act

    # Swing trading specifics
    max_holding_days: int = 14           # Max days to hold a position
    min_holding_days: int = 1            # Min days before exit allowed
