"""Tests for paper trading engine."""
import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.engine.paper_trader import PaperTradingEngine, OrderSide, OrderStatus
from config.trading_config import TradingConfig


@pytest.fixture
def engine():
    """Create a fresh paper trading engine."""
    config = TradingConfig(initial_capital=10000.0)
    return PaperTradingEngine(config)


class TestPaperTradingEngine:
    """Test suite for PaperTradingEngine."""

    def test_initial_state(self, engine):
        """Test engine initializes with correct state."""
        assert engine.cash_balance == 10000.0
        assert engine.get_portfolio_value() == 10000.0
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0

    def test_buy_order(self, engine):
        """Test executing a buy order."""
        timestamp = datetime.now()
        order = engine.submit_order(
            symbol="bitcoin",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0,
            timestamp=timestamp
        )

        assert order.status == OrderStatus.FILLED
        assert "bitcoin" in engine.positions
        assert engine.positions["bitcoin"].quantity == 0.1
        assert engine.cash_balance < 10000.0  # Should have spent money

    def test_sell_order(self, engine):
        """Test executing a sell order."""
        timestamp = datetime.now()

        # First buy
        engine.submit_order("bitcoin", OrderSide.BUY, 0.1, 50000.0, timestamp)

        # Then sell
        order = engine.submit_order(
            symbol="bitcoin",
            side=OrderSide.SELL,
            quantity=0.1,
            price=55000.0,  # Profit scenario
            timestamp=timestamp + timedelta(days=1)
        )

        assert order.status == OrderStatus.FILLED
        assert "bitcoin" not in engine.positions  # Position closed
        assert len(engine.trades) == 1  # Trade recorded
        assert engine.trades[0]["pnl"] > 0  # Profitable trade

    def test_insufficient_funds(self, engine):
        """Test order rejection when insufficient funds."""
        timestamp = datetime.now()
        order = engine.submit_order(
            symbol="bitcoin",
            side=OrderSide.BUY,
            quantity=100,  # Way too much
            price=50000.0,
            timestamp=timestamp
        )

        assert order.status == OrderStatus.REJECTED
        assert "Insufficient funds" in order.rejection_reason

    def test_sell_without_position(self, engine):
        """Test sell order rejected when no position exists."""
        timestamp = datetime.now()
        order = engine.submit_order(
            symbol="bitcoin",
            side=OrderSide.SELL,
            quantity=0.1,
            price=50000.0,
            timestamp=timestamp
        )

        assert order.status == OrderStatus.REJECTED
        assert "No position" in order.rejection_reason

    def test_portfolio_value_tracking(self, engine):
        """Test portfolio value updates correctly."""
        timestamp = datetime.now()

        # Buy position
        engine.submit_order("bitcoin", OrderSide.BUY, 0.1, 50000.0, timestamp)

        # Update price
        engine.update_prices({"bitcoin": 60000.0}, timestamp)

        # Portfolio should reflect new value
        portfolio_value = engine.get_portfolio_value()
        # Cash spent + position value at new price
        assert portfolio_value > 10000.0  # Should have gained value

    def test_fees_applied(self, engine):
        """Test trading fees are deducted."""
        initial_cash = engine.cash_balance
        timestamp = datetime.now()

        order = engine.submit_order("bitcoin", OrderSide.BUY, 0.1, 50000.0, timestamp)

        expected_cost = 0.1 * 50000.0 * (1 + engine.config.slippage_pct)
        expected_fees = expected_cost * engine.config.trading_fee_pct

        assert order.fees > 0
        assert engine.cash_balance < initial_cash - (0.1 * 50000.0)  # Cost + fees

    def test_reset(self, engine):
        """Test engine reset to initial state."""
        timestamp = datetime.now()
        engine.submit_order("bitcoin", OrderSide.BUY, 0.1, 50000.0, timestamp)

        engine.reset()

        assert engine.cash_balance == 10000.0
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
