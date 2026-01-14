"""Coin registry with category classifications."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class CoinCategory(Enum):
    """Cryptocurrency categories for diversified trading."""
    LARGE_CAP = "large_cap"           # BTC, ETH - most stable
    ALTCOIN = "altcoin"               # Major altcoins
    DEFI = "defi"                     # DeFi tokens
    LAYER2 = "layer2"                 # L2 scaling solutions
    MEMECOIN = "memecoin"             # Meme coins - high risk
    AI = "ai"                         # AI-related tokens
    GAMING = "gaming"                 # Gaming/Metaverse
    STABLECOIN = "stablecoin"         # Stablecoins (for reference only)


@dataclass
class CoinInfo:
    """Information about a cryptocurrency."""
    id: str                           # CoinGecko ID
    symbol: str                       # Ticker symbol
    name: str                         # Full name
    category: CoinCategory            # Category classification
    market_cap_rank: int = 0          # Market cap ranking
    volatility_tier: str = "medium"   # low, medium, high, extreme
    enabled: bool = True              # Whether to trade this coin
    max_position_pct: float = 0.20    # Max position size override
    notes: str = ""                   # Additional notes


@dataclass
class CategoryConfig:
    """Configuration for a coin category."""
    category: CoinCategory
    enabled: bool = True
    max_allocation_pct: float = 0.25      # Max % of portfolio for this category
    max_coins: int = 3                     # Max coins to hold from this category
    position_size_multiplier: float = 1.0  # Adjust position sizes
    notes: str = ""


class CoinRegistry:
    """
    Registry of tradeable cryptocurrencies organized by category.

    Provides diversified exposure across different crypto sectors.
    """

    def __init__(self):
        self.coins: Dict[str, CoinInfo] = {}
        self.category_configs: Dict[CoinCategory, CategoryConfig] = {}
        self._initialize_coins()
        self._initialize_category_configs()

    def _initialize_coins(self):
        """Initialize the coin registry with categorized coins."""

        # Large Cap - Most stable, lower volatility
        large_caps = [
            CoinInfo("bitcoin", "BTC", "Bitcoin", CoinCategory.LARGE_CAP,
                    1, "medium", True, 0.30, "Digital gold, store of value"),
            CoinInfo("ethereum", "ETH", "Ethereum", CoinCategory.LARGE_CAP,
                    2, "medium", True, 0.25, "Smart contract platform"),
        ]

        # Major Altcoins
        altcoins = [
            CoinInfo("binancecoin", "BNB", "BNB", CoinCategory.ALTCOIN,
                    4, "medium", True, 0.15, "Binance ecosystem"),
            CoinInfo("solana", "SOL", "Solana", CoinCategory.ALTCOIN,
                    5, "high", True, 0.15, "High-speed blockchain"),
            CoinInfo("ripple", "XRP", "XRP", CoinCategory.ALTCOIN,
                    6, "high", True, 0.15, "Cross-border payments"),
            CoinInfo("cardano", "ADA", "Cardano", CoinCategory.ALTCOIN,
                    8, "high", True, 0.12, "Proof-of-stake platform"),
            CoinInfo("avalanche-2", "AVAX", "Avalanche", CoinCategory.ALTCOIN,
                    12, "high", True, 0.12, "Fast finality blockchain"),
            CoinInfo("polkadot", "DOT", "Polkadot", CoinCategory.ALTCOIN,
                    13, "high", True, 0.12, "Interoperability protocol"),
            CoinInfo("chainlink", "LINK", "Chainlink", CoinCategory.ALTCOIN,
                    14, "high", True, 0.12, "Oracle network"),
            CoinInfo("litecoin", "LTC", "Litecoin", CoinCategory.ALTCOIN,
                    20, "medium", True, 0.10, "Digital silver"),
        ]

        # DeFi Tokens
        defi_tokens = [
            CoinInfo("uniswap", "UNI", "Uniswap", CoinCategory.DEFI,
                    22, "high", True, 0.10, "Leading DEX"),
            CoinInfo("aave", "AAVE", "Aave", CoinCategory.DEFI,
                    35, "high", True, 0.10, "Lending protocol"),
            CoinInfo("maker", "MKR", "Maker", CoinCategory.DEFI,
                    45, "high", True, 0.10, "DAI stablecoin governance"),
            CoinInfo("compound-governance-token", "COMP", "Compound", CoinCategory.DEFI,
                    90, "high", True, 0.08, "Lending protocol"),
            CoinInfo("curve-dao-token", "CRV", "Curve", CoinCategory.DEFI,
                    95, "high", True, 0.08, "Stablecoin DEX"),
        ]

        # Layer 2 Solutions
        layer2_tokens = [
            CoinInfo("matic-network", "MATIC", "Polygon", CoinCategory.LAYER2,
                    15, "high", True, 0.12, "Ethereum L2"),
            CoinInfo("arbitrum", "ARB", "Arbitrum", CoinCategory.LAYER2,
                    40, "high", True, 0.10, "Ethereum rollup"),
            CoinInfo("optimism", "OP", "Optimism", CoinCategory.LAYER2,
                    42, "high", True, 0.10, "Ethereum rollup"),
            CoinInfo("immutable-x", "IMX", "Immutable X", CoinCategory.LAYER2,
                    50, "high", True, 0.08, "NFT L2"),
        ]

        # Meme Coins - High risk, high volatility
        memecoins = [
            CoinInfo("dogecoin", "DOGE", "Dogecoin", CoinCategory.MEMECOIN,
                    9, "extreme", True, 0.08, "Original memecoin"),
            CoinInfo("shiba-inu", "SHIB", "Shiba Inu", CoinCategory.MEMECOIN,
                    11, "extreme", True, 0.06, "DOGE competitor"),
            CoinInfo("pepe", "PEPE", "Pepe", CoinCategory.MEMECOIN,
                    25, "extreme", True, 0.05, "Frog memecoin"),
            CoinInfo("floki", "FLOKI", "Floki", CoinCategory.MEMECOIN,
                    55, "extreme", True, 0.05, "Viking memecoin"),
            CoinInfo("bonk", "BONK", "Bonk", CoinCategory.MEMECOIN,
                    60, "extreme", True, 0.05, "Solana memecoin"),
        ]

        # AI Tokens
        ai_tokens = [
            CoinInfo("render-token", "RNDR", "Render", CoinCategory.AI,
                    30, "high", True, 0.10, "GPU rendering"),
            CoinInfo("fetch-ai", "FET", "Fetch.ai", CoinCategory.AI,
                    48, "high", True, 0.08, "AI agents"),
            CoinInfo("singularitynet", "AGIX", "SingularityNET", CoinCategory.AI,
                    85, "high", True, 0.08, "AI marketplace"),
            CoinInfo("ocean-protocol", "OCEAN", "Ocean Protocol", CoinCategory.AI,
                    120, "high", True, 0.06, "Data marketplace"),
        ]

        # Gaming/Metaverse
        gaming_tokens = [
            CoinInfo("the-sandbox", "SAND", "The Sandbox", CoinCategory.GAMING,
                    52, "high", True, 0.08, "Metaverse gaming"),
            CoinInfo("decentraland", "MANA", "Decentraland", CoinCategory.GAMING,
                    65, "high", True, 0.08, "Virtual world"),
            CoinInfo("axie-infinity", "AXS", "Axie Infinity", CoinCategory.GAMING,
                    70, "high", True, 0.08, "Play-to-earn"),
            CoinInfo("gala", "GALA", "Gala", CoinCategory.GAMING,
                    75, "high", True, 0.06, "Gaming platform"),
        ]

        # Stablecoins (for reference/holding, not trading)
        stablecoins = [
            CoinInfo("tether", "USDT", "Tether", CoinCategory.STABLECOIN,
                    3, "low", False, 0.0, "USD stablecoin"),
            CoinInfo("usd-coin", "USDC", "USD Coin", CoinCategory.STABLECOIN,
                    7, "low", False, 0.0, "USD stablecoin"),
            CoinInfo("dai", "DAI", "Dai", CoinCategory.STABLECOIN,
                    24, "low", False, 0.0, "Decentralized stablecoin"),
        ]

        # Add all coins to registry
        all_coins = (large_caps + altcoins + defi_tokens + layer2_tokens +
                    memecoins + ai_tokens + gaming_tokens + stablecoins)

        for coin in all_coins:
            self.coins[coin.id] = coin

    def _initialize_category_configs(self):
        """Initialize category trading configurations."""

        self.category_configs = {
            CoinCategory.LARGE_CAP: CategoryConfig(
                CoinCategory.LARGE_CAP,
                enabled=True,
                max_allocation_pct=0.40,  # Up to 40% in large caps
                max_coins=2,
                position_size_multiplier=1.2,  # Larger positions allowed
                notes="Core holdings, most stable"
            ),
            CoinCategory.ALTCOIN: CategoryConfig(
                CoinCategory.ALTCOIN,
                enabled=True,
                max_allocation_pct=0.30,
                max_coins=4,
                position_size_multiplier=1.0,
                notes="Major altcoins for growth"
            ),
            CoinCategory.DEFI: CategoryConfig(
                CoinCategory.DEFI,
                enabled=True,
                max_allocation_pct=0.15,
                max_coins=2,
                position_size_multiplier=0.8,
                notes="DeFi exposure"
            ),
            CoinCategory.LAYER2: CategoryConfig(
                CoinCategory.LAYER2,
                enabled=True,
                max_allocation_pct=0.15,
                max_coins=2,
                position_size_multiplier=0.8,
                notes="Scaling solutions"
            ),
            CoinCategory.MEMECOIN: CategoryConfig(
                CoinCategory.MEMECOIN,
                enabled=True,
                max_allocation_pct=0.10,  # Limit meme exposure
                max_coins=2,
                position_size_multiplier=0.5,  # Smaller positions
                notes="High risk, speculative"
            ),
            CoinCategory.AI: CategoryConfig(
                CoinCategory.AI,
                enabled=True,
                max_allocation_pct=0.10,
                max_coins=2,
                position_size_multiplier=0.7,
                notes="AI narrative"
            ),
            CoinCategory.GAMING: CategoryConfig(
                CoinCategory.GAMING,
                enabled=True,
                max_allocation_pct=0.10,
                max_coins=2,
                position_size_multiplier=0.7,
                notes="Gaming/Metaverse sector"
            ),
            CoinCategory.STABLECOIN: CategoryConfig(
                CoinCategory.STABLECOIN,
                enabled=False,  # Don't trade stablecoins
                max_allocation_pct=0.0,
                max_coins=0,
                position_size_multiplier=0.0,
                notes="Reference only"
            ),
        }

    def get_coin(self, coin_id: str) -> Optional[CoinInfo]:
        """Get coin info by ID."""
        return self.coins.get(coin_id)

    def get_coins_by_category(self, category: CoinCategory) -> List[CoinInfo]:
        """Get all coins in a category."""
        return [
            coin for coin in self.coins.values()
            if coin.category == category and coin.enabled
        ]

    def get_all_tradeable_coins(self) -> List[CoinInfo]:
        """Get all enabled tradeable coins."""
        return [
            coin for coin in self.coins.values()
            if coin.enabled and self.category_configs[coin.category].enabled
        ]

    def get_coins_by_volatility(self, tier: str) -> List[CoinInfo]:
        """Get coins by volatility tier (low, medium, high, extreme)."""
        return [
            coin for coin in self.coins.values()
            if coin.volatility_tier == tier and coin.enabled
        ]

    def get_category_config(self, category: CoinCategory) -> CategoryConfig:
        """Get configuration for a category."""
        return self.category_configs[category]

    def get_enabled_categories(self) -> List[CoinCategory]:
        """Get list of enabled trading categories."""
        return [
            cat for cat, config in self.category_configs.items()
            if config.enabled
        ]

    def get_coin_ids(self) -> List[str]:
        """Get all tradeable coin IDs for data fetching."""
        return [coin.id for coin in self.get_all_tradeable_coins()]

    def get_summary(self) -> Dict:
        """Get summary of registered coins."""
        summary = {
            "total_coins": len(self.coins),
            "tradeable_coins": len(self.get_all_tradeable_coins()),
            "by_category": {}
        }

        for category in CoinCategory:
            coins = self.get_coins_by_category(category)
            config = self.category_configs[category]
            summary["by_category"][category.value] = {
                "count": len(coins),
                "enabled": config.enabled,
                "max_allocation": config.max_allocation_pct,
                "coins": [c.symbol for c in coins[:5]]  # First 5
            }

        return summary


# Global registry instance
coin_registry = CoinRegistry()


def get_coin_registry() -> CoinRegistry:
    """Get the global coin registry."""
    return coin_registry
