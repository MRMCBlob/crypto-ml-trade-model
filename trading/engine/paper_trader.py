"""Paper trading simulation engine."""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config.trading_config import TradingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order representation."""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    fees: float = 0.0
    rejection_reason: Optional[str] = None


@dataclass
class Position:
    """Open position representation."""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_price(self, current_price: float):
        """Update position with current market price."""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = (current_price / self.entry_price - 1) * 100


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    cash_balance: float
    positions_value: float
    total_value: float
    positions: Dict[str, Dict[str, Any]]
    daily_return: float = 0.0


class PaperTradingEngine:
    """
    Simulated trading engine for paper trading.

    Tracks portfolio, executes simulated trades, and calculates performance.
    Includes realistic slippage and fee simulation.
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize paper trading engine.

        Args:
            config: Trading configuration
        """
        self.config = config or TradingConfig()
        self.initial_capital = self.config.initial_capital
        self.cash_balance = self.initial_capital

        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        self.snapshots: List[PortfolioSnapshot] = []

        logger.info(f"Paper trading engine initialized with ${self.initial_capital:,.2f}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"ORD-{uuid.uuid4().hex[:8].upper()}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply simulated slippage to execution price."""
        if side == OrderSide.BUY:
            return price * (1 + self.config.slippage_pct)
        else:
            return price * (1 - self.config.slippage_pct)

    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees."""
        return quantity * price * self.config.trading_fee_pct

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash_balance + positions_value

    def get_positions_value(self) -> float:
        """Get total value of all positions."""
        return sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )

    def get_exposure(self) -> float:
        """Get current portfolio exposure as percentage."""
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return 0.0
        return self.get_positions_value() / total_value

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: datetime
    ) -> Order:
        """
        Submit a new order for execution.

        In paper trading, orders are filled immediately at simulated price.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Amount to trade
            price: Current market price
            timestamp: Order timestamp

        Returns:
            Executed order
        """
        order = Order(
            id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Simulate execution
        execution_price = self._apply_slippage(price, side)
        fees = self._calculate_fees(quantity, execution_price)

        if side == OrderSide.BUY:
            total_cost = (quantity * execution_price) + fees

            # Check if we have enough cash
            if total_cost > self.cash_balance:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash_balance:.2f}"
                self.orders.append(order)
                logger.warning(f"Order rejected: {order.rejection_reason}")
                return order

            # Check exposure limit
            new_exposure = (self.get_positions_value() + quantity * execution_price) / self.get_portfolio_value()
            if new_exposure > self.config.max_total_exposure:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"Exposure limit exceeded: {new_exposure:.1%} > {self.config.max_total_exposure:.1%}"
                self.orders.append(order)
                logger.warning(f"Order rejected: {order.rejection_reason}")
                return order

            # Execute buy
            self.cash_balance -= total_cost

            if symbol in self.positions:
                # Add to existing position (average in)
                existing = self.positions[symbol]
                new_quantity = existing.quantity + quantity
                new_avg_price = (
                    (existing.quantity * existing.entry_price) +
                    (quantity * execution_price)
                ) / new_quantity
                existing.quantity = new_quantity
                existing.entry_price = new_avg_price
                existing.update_price(execution_price)
            else:
                # Open new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    entry_timestamp=timestamp,
                    current_price=execution_price
                )

            logger.info(
                f"BUY {quantity:.6f} {symbol} @ ${execution_price:.2f} "
                f"(fees: ${fees:.2f})"
            )

        else:  # SELL
            if symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"No position in {symbol}"
                self.orders.append(order)
                logger.warning(f"Order rejected: {order.rejection_reason}")
                return order

            position = self.positions[symbol]
            sell_quantity = min(quantity, position.quantity)

            # Execute sell
            proceeds = (sell_quantity * execution_price) - fees
            self.cash_balance += proceeds

            # Calculate realized PnL
            realized_pnl = (execution_price - position.entry_price) * sell_quantity

            # Record trade
            self.trades.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side.value,
                "quantity": sell_quantity,
                "entry_price": position.entry_price,
                "exit_price": execution_price,
                "pnl": realized_pnl,
                "pnl_pct": (execution_price / position.entry_price - 1) * 100,
                "fees": fees,
                "holding_days": (timestamp - position.entry_timestamp).days
            })

            # Update or close position
            position.quantity -= sell_quantity
            if position.quantity <= 0.0001:  # Effectively zero
                del self.positions[symbol]
                logger.info(
                    f"SELL (close) {sell_quantity:.6f} {symbol} @ ${execution_price:.2f} "
                    f"PnL: ${realized_pnl:.2f} ({(execution_price / position.entry_price - 1) * 100:.2f}%)"
                )
            else:
                logger.info(
                    f"SELL (partial) {sell_quantity:.6f} {symbol} @ ${execution_price:.2f} "
                    f"PnL: ${realized_pnl:.2f}"
                )

        order.status = OrderStatus.FILLED
        order.filled_price = execution_price
        order.filled_timestamp = timestamp
        order.fees = fees
        self.orders.append(order)

        return order

    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """
        Update all position prices and record snapshot.

        Args:
            prices: Dictionary of symbol -> current price
            timestamp: Current timestamp
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        self._record_snapshot(timestamp)

    def _record_snapshot(self, timestamp: datetime):
        """Record current portfolio state."""
        positions_value = self.get_positions_value()
        total_value = self.cash_balance + positions_value

        # Calculate daily return
        daily_return = 0.0
        if len(self.snapshots) > 0:
            prev_value = self.snapshots[-1].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value

        # Serialize positions
        positions_dict = {
            symbol: {
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct
            }
            for symbol, pos in self.positions.items()
        }

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_value=total_value,
            positions=positions_dict,
            daily_return=daily_return
        )

        self.snapshots.append(snapshot)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol if it exists."""
        return self.positions.get(symbol)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get all completed trades."""
        return self.trades

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        total_value = self.get_portfolio_value()
        total_return = (total_value / self.initial_capital - 1) * 100

        total_pnl = sum(t["pnl"] for t in self.trades)
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] < 0]

        return {
            "initial_capital": self.initial_capital,
            "cash_balance": self.cash_balance,
            "positions_value": self.get_positions_value(),
            "total_value": total_value,
            "total_return_pct": total_return,
            "exposure": self.get_exposure(),
            "num_positions": len(self.positions),
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) if self.trades else 0,
            "total_realized_pnl": total_pnl,
            "avg_trade_pnl": total_pnl / len(self.trades) if self.trades else 0
        }

    def reset(self):
        """Reset engine to initial state."""
        self.cash_balance = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.snapshots = []
        logger.info("Paper trading engine reset")
