"""Multi-coin portfolio management with category allocation."""
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from config.coin_registry import (
    CoinRegistry, CoinInfo, CoinCategory, CategoryConfig,
    get_coin_registry
)
from config.trading_config import TradingConfig
from .paper_trader import PaperTradingEngine, Position, OrderSide
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CategoryAllocation:
    """Current allocation for a category."""
    category: CoinCategory
    target_allocation: float
    current_allocation: float
    current_value: float
    positions: List[str]  # Coin IDs in this category
    available_allocation: float = 0.0


@dataclass
class PortfolioState:
    """Current state of the multi-coin portfolio."""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    num_positions: int
    allocations: Dict[CoinCategory, CategoryAllocation]
    positions: Dict[str, Dict[str, Any]]


class MultiCoinPortfolioManager:
    """
    Manages a diversified multi-coin portfolio with category-based allocation.

    Ensures diversification across different crypto sectors while respecting
    allocation limits per category.
    """

    def __init__(
        self,
        engine: PaperTradingEngine,
        registry: Optional[CoinRegistry] = None,
        config: Optional[TradingConfig] = None
    ):
        """
        Initialize portfolio manager.

        Args:
            engine: Paper trading engine for order execution
            registry: Coin registry with category info
            config: Trading configuration
        """
        self.engine = engine
        self.registry = registry or get_coin_registry()
        self.config = config or TradingConfig()

        # Track which coins have models trained
        self.trained_models: Dict[str, bool] = {}

        # Category position counts
        self.category_positions: Dict[CoinCategory, List[str]] = {
            cat: [] for cat in CoinCategory
        }

        logger.info("Multi-coin portfolio manager initialized")

    def get_portfolio_state(self, prices: Dict[str, float]) -> PortfolioState:
        """
        Get current portfolio state with category allocations.

        Args:
            prices: Current prices for all coins

        Returns:
            PortfolioState with detailed allocation info
        """
        # Update positions with current prices
        self.engine.update_prices(prices, datetime.now())

        total_value = self.engine.get_portfolio_value()
        positions_value = self.engine.get_positions_value()

        # Calculate category allocations
        allocations = {}
        self.category_positions = {cat: [] for cat in CoinCategory}

        for coin_id, position in self.engine.positions.items():
            coin_info = self.registry.get_coin(coin_id)
            if coin_info:
                self.category_positions[coin_info.category].append(coin_id)

        for category in CoinCategory:
            cat_config = self.registry.get_category_config(category)
            cat_positions = self.category_positions[category]

            # Calculate current value in this category
            cat_value = sum(
                self.engine.positions[coin_id].quantity *
                self.engine.positions[coin_id].current_price
                for coin_id in cat_positions
                if coin_id in self.engine.positions
            )

            current_alloc = cat_value / total_value if total_value > 0 else 0
            available = max(0, cat_config.max_allocation_pct - current_alloc)

            allocations[category] = CategoryAllocation(
                category=category,
                target_allocation=cat_config.max_allocation_pct,
                current_allocation=current_alloc,
                current_value=cat_value,
                positions=cat_positions,
                available_allocation=available
            )

        # Serialize positions
        positions_dict = {
            coin_id: {
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "category": self.registry.get_coin(coin_id).category.value
                           if self.registry.get_coin(coin_id) else "unknown"
            }
            for coin_id, pos in self.engine.positions.items()
        }

        return PortfolioState(
            timestamp=datetime.now(),
            total_value=total_value,
            cash_balance=self.engine.cash_balance,
            positions_value=positions_value,
            num_positions=len(self.engine.positions),
            allocations=allocations,
            positions=positions_dict
        )

    def can_open_position(
        self,
        coin_id: str,
        position_value: float,
        prices: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened for a coin.

        Args:
            coin_id: Coin to potentially buy
            position_value: Proposed position value
            prices: Current prices

        Returns:
            Tuple of (allowed, reason)
        """
        coin_info = self.registry.get_coin(coin_id)
        if not coin_info:
            return False, f"Coin {coin_id} not in registry"

        if not coin_info.enabled:
            return False, f"Coin {coin_id} is disabled"

        category = coin_info.category
        cat_config = self.registry.get_category_config(category)

        if not cat_config.enabled:
            return False, f"Category {category.value} is disabled"

        state = self.get_portfolio_state(prices)

        # Check category position limit
        current_cat_positions = len(self.category_positions[category])
        if current_cat_positions >= cat_config.max_coins:
            return False, f"Category {category.value} at max positions ({cat_config.max_coins})"

        # Check category allocation limit
        cat_alloc = state.allocations[category]
        new_alloc = (cat_alloc.current_value + position_value) / state.total_value

        if new_alloc > cat_config.max_allocation_pct:
            return False, (
                f"Would exceed category allocation: "
                f"{new_alloc:.1%} > {cat_config.max_allocation_pct:.1%}"
            )

        # Check individual coin position limit
        coin_max = coin_info.max_position_pct * cat_config.position_size_multiplier
        coin_alloc = position_value / state.total_value

        if coin_alloc > coin_max:
            return False, f"Position too large: {coin_alloc:.1%} > {coin_max:.1%}"

        # Check total exposure
        new_exposure = (state.positions_value + position_value) / state.total_value
        if new_exposure > self.config.max_total_exposure:
            return False, f"Would exceed total exposure: {new_exposure:.1%}"

        return True, "OK"

    def calculate_position_size(
        self,
        coin_id: str,
        current_price: float,
        atr: float,
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate position size for a coin respecting category limits.

        Args:
            coin_id: Coin to buy
            current_price: Current price
            atr: Average True Range
            prices: All current prices

        Returns:
            Position size in units
        """
        coin_info = self.registry.get_coin(coin_id)
        if not coin_info:
            return 0.0

        cat_config = self.registry.get_category_config(coin_info.category)
        state = self.get_portfolio_state(prices)

        # Base risk-based sizing
        risk_amount = state.total_value * self.config.risk_per_trade_pct
        stop_distance = atr * self.config.stop_loss_atr_multiple

        if stop_distance <= 0:
            return 0.0

        risk_based_qty = risk_amount / stop_distance

        # Coin-specific max
        coin_max_value = state.total_value * coin_info.max_position_pct
        coin_max_qty = coin_max_value / current_price

        # Category allocation limit
        cat_alloc = state.allocations[coin_info.category]
        cat_available_value = cat_alloc.available_allocation * state.total_value
        cat_max_qty = cat_available_value / current_price

        # Apply category multiplier
        adjusted_qty = min(risk_based_qty, coin_max_qty, cat_max_qty)
        adjusted_qty *= cat_config.position_size_multiplier

        # Ensure we don't exceed cash
        max_affordable = (state.cash_balance * 0.95) / current_price  # Keep 5% buffer
        final_qty = min(adjusted_qty, max_affordable)

        logger.debug(
            f"Position sizing for {coin_id}: "
            f"risk={risk_based_qty:.6f}, coin_max={coin_max_qty:.6f}, "
            f"cat_max={cat_max_qty:.6f}, final={final_qty:.6f}"
        )

        return max(0, final_qty)

    def get_rebalance_suggestions(
        self,
        prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions for portfolio rebalancing.

        Returns:
            List of suggested actions (buy/sell recommendations)
        """
        state = self.get_portfolio_state(prices)
        suggestions = []

        for category, alloc in state.allocations.items():
            cat_config = self.registry.get_category_config(category)

            if not cat_config.enabled:
                continue

            # Check if overallocated
            if alloc.current_allocation > cat_config.max_allocation_pct * 1.1:  # 10% buffer
                excess = alloc.current_allocation - cat_config.max_allocation_pct
                suggestions.append({
                    "action": "reduce",
                    "category": category.value,
                    "reason": f"Overallocated by {excess:.1%}",
                    "positions": alloc.positions
                })

            # Check if significantly underallocated
            elif (alloc.current_allocation < cat_config.max_allocation_pct * 0.5 and
                  len(alloc.positions) < cat_config.max_coins):
                gap = cat_config.max_allocation_pct - alloc.current_allocation
                suggestions.append({
                    "action": "increase",
                    "category": category.value,
                    "reason": f"Underallocated by {gap:.1%}",
                    "available_coins": [
                        c.id for c in self.registry.get_coins_by_category(category)
                        if c.id not in alloc.positions
                    ][:3]
                })

        return suggestions

    def print_portfolio_summary(self, prices: Dict[str, float]):
        """Print formatted portfolio summary."""
        state = self.get_portfolio_state(prices)

        print("\n" + "=" * 70)
        print("MULTI-COIN PORTFOLIO SUMMARY")
        print("=" * 70)

        print(f"\nTotal Value:      ${state.total_value:,.2f}")
        print(f"Cash Balance:     ${state.cash_balance:,.2f}")
        print(f"Positions Value:  ${state.positions_value:,.2f}")
        print(f"Num Positions:    {state.num_positions}")

        print("\n--- Category Allocations ---")
        for category, alloc in state.allocations.items():
            if alloc.current_value > 0 or self.registry.get_category_config(category).enabled:
                status = "ENABLED" if self.registry.get_category_config(category).enabled else "disabled"
                print(
                    f"  {category.value:12} | "
                    f"Current: {alloc.current_allocation:5.1%} | "
                    f"Target: {alloc.target_allocation:5.1%} | "
                    f"Value: ${alloc.current_value:>10,.2f} | "
                    f"Positions: {len(alloc.positions)} | {status}"
                )

        if state.positions:
            print("\n--- Open Positions ---")
            for coin_id, pos in state.positions.items():
                coin = self.registry.get_coin(coin_id)
                symbol = coin.symbol if coin else coin_id.upper()
                pnl_sign = "+" if pos["unrealized_pnl"] >= 0 else ""
                print(
                    f"  {symbol:8} | "
                    f"Qty: {pos['quantity']:.6f} | "
                    f"Entry: ${pos['entry_price']:>10,.2f} | "
                    f"Current: ${pos['current_price']:>10,.2f} | "
                    f"P&L: {pnl_sign}${pos['unrealized_pnl']:>8,.2f} ({pnl_sign}{pos['unrealized_pnl_pct']:.1f}%) | "
                    f"[{pos['category']}]"
                )

        print("=" * 70 + "\n")
