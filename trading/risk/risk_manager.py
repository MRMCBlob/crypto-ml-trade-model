"""Risk management for trading operations."""
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config.trading_config import TradingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_consecutive_losses: int = 5
    max_positions: int = 5


class RiskManager:
    """
    Real-time risk management for trading operations.

    Monitors portfolio risk and enforces limits to prevent
    catastrophic losses.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize risk manager.

        Args:
            config: Trading configuration with risk limits
        """
        self.config = config or TradingConfig()
        self.limits = RiskLimits(
            max_daily_loss_pct=self.config.max_daily_loss_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_consecutive_losses=self.config.max_consecutive_losses,
            max_positions=self.config.max_positions
        )

        self.daily_start_value: Optional[float] = None
        self.peak_value: Optional[float] = None
        self.consecutive_losses: int = 0
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.halt_timestamp: Optional[datetime] = None

    def update_daily_start(self, portfolio_value: float):
        """
        Set daily starting value for daily loss tracking.

        Call this at the start of each trading day.
        """
        self.daily_start_value = portfolio_value
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        logger.debug(f"Daily start value set to ${portfolio_value:,.2f}")

    def update_peak(self, portfolio_value: float):
        """Update peak value if current value is higher."""
        if self.peak_value is None or portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

    def check_risk_limits(
        self,
        current_value: float,
        num_positions: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all risk limits.

        Args:
            current_value: Current portfolio value
            num_positions: Number of open positions

        Returns:
            Tuple of (is_trading_allowed, violation_message)
        """
        if self.trading_halted:
            return False, self.halt_reason

        # Daily loss check
        if self.daily_start_value:
            daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
            if daily_loss >= self.limits.max_daily_loss_pct:
                self._halt_trading(
                    f"Daily loss limit reached: {daily_loss:.2%} >= {self.limits.max_daily_loss_pct:.2%}"
                )
                return False, self.halt_reason

        # Drawdown check
        if self.peak_value:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown >= self.limits.max_drawdown_pct:
                self._halt_trading(
                    f"Max drawdown reached: {drawdown:.2%} >= {self.limits.max_drawdown_pct:.2%}"
                )
                return False, self.halt_reason

        # Consecutive losses check
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            self._halt_trading(
                f"Consecutive losses limit: {self.consecutive_losses} >= {self.limits.max_consecutive_losses}"
            )
            return False, self.halt_reason

        # Position count check (warning, not halt)
        if num_positions >= self.limits.max_positions:
            return False, f"Max positions reached: {num_positions} >= {self.limits.max_positions}"

        return True, None

    def record_trade_result(self, pnl: float):
        """
        Record trade result for consecutive loss tracking.

        Args:
            pnl: Profit/loss from the trade
        """
        if pnl < 0:
            self.consecutive_losses += 1
            logger.debug(f"Consecutive losses: {self.consecutive_losses}")
        else:
            if self.consecutive_losses > 0:
                logger.debug(f"Winning trade, resetting consecutive losses from {self.consecutive_losses}")
            self.consecutive_losses = 0

    def _halt_trading(self, reason: str):
        """Halt all trading operations."""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.now()
        logger.error(f"TRADING HALTED: {reason}")

    def resume_trading(self):
        """Resume trading after manual review."""
        if self.trading_halted:
            logger.warning(
                f"Trading resumed. Was halted at {self.halt_timestamp} for: {self.halt_reason}"
            )
        self.trading_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        self.consecutive_losses = 0

    def get_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_value is None or self.peak_value == 0:
            return 0.0
        return (self.peak_value - current_value) / self.peak_value

    def get_daily_pnl(self, current_value: float) -> Tuple[float, float]:
        """
        Get daily P&L in dollars and percentage.

        Returns:
            Tuple of (pnl_dollars, pnl_percentage)
        """
        if self.daily_start_value is None:
            return 0.0, 0.0

        pnl_dollars = current_value - self.daily_start_value
        pnl_pct = pnl_dollars / self.daily_start_value
        return pnl_dollars, pnl_pct

    def get_status(self, current_value: float, num_positions: int) -> dict:
        """Get comprehensive risk status."""
        daily_pnl, daily_pnl_pct = self.get_daily_pnl(current_value)
        drawdown = self.get_current_drawdown(current_value)

        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "current_value": current_value,
            "peak_value": self.peak_value,
            "daily_start_value": self.daily_start_value,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "current_drawdown": drawdown,
            "max_drawdown_limit": self.limits.max_drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.limits.max_consecutive_losses,
            "num_positions": num_positions,
            "max_positions": self.limits.max_positions
        }
