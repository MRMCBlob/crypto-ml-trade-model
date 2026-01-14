# Crypto ML Trading System

A Python-based machine learning system for cryptocurrency swing trading across multiple coins and categories. Uses a hybrid LSTM + XGBoost model for price direction prediction with paper trading simulation.

## Features

- **Multi-Coin Trading**: Trade 30+ cryptocurrencies across different categories
- **Category-Based Allocation**: Diversified exposure to Large Cap, Altcoins, DeFi, Layer 2, Memecoins, AI, and Gaming tokens
- **Hybrid ML Model**: Combines LSTM (temporal patterns) + XGBoost (technical indicators)
- **50+ Technical Indicators**: Trend, momentum, volatility, and volume indicators
- **Paper Trading Engine**: Simulates realistic trading with slippage and fees
- **Risk Management**: Position sizing, drawdown limits, category allocation, and circuit breakers
- **Docker Support**: Containerized deployment for easy setup

## Coin Categories

| Category | Description | Max Allocation | Example Coins |
|----------|-------------|----------------|---------------|
| Large Cap | BTC, ETH - most stable | 40% | BTC, ETH |
| Altcoin | Major altcoins | 30% | SOL, ADA, AVAX, DOT |
| DeFi | DeFi protocols | 15% | UNI, AAVE, MKR |
| Layer 2 | L2 scaling solutions | 15% | MATIC, ARB, OP |
| Memecoin | High risk meme coins | 10% | DOGE, SHIB, PEPE |
| AI | AI-related tokens | 10% | RNDR, FET, AGIX |
| Gaming | Gaming/Metaverse | 10% | SAND, MANA, AXS |

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d crypto-trader

# Or run individual services
docker-compose --profile data up data-fetcher       # Download data
docker-compose --profile training up model-trainer  # Train models
docker-compose --profile backtest up backtester     # Run backtest
docker-compose up crypto-trader                      # Run trading
```

### Option 2: Local Installation

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Download Data for All Coins**
```bash
python scripts/download_multi_coin_data.py
```

**3. Train Models for All Coins**
```bash
python scripts/train_multi_coin_models.py
```

**4. Run Backtest**
```bash
python scripts/run_multi_coin_backtest.py
```

**5. Start Paper Trading**
```bash
# Single cycle
python scripts/run_multi_coin_trading.py

# Continuous (daily cycles)
python scripts/run_multi_coin_trading.py --continuous
```

## Project Structure

```
crypto-ml/
├── config/
│   ├── settings.py          # Global settings
│   ├── trading_config.py    # Trading parameters
│   ├── model_config.py      # ML model config
│   └── coin_registry.py     # Coin categories & allocation
├── data/
│   ├── fetchers/            # CoinGecko API fetcher
│   ├── processors/          # Feature engineering
│   └── storage/             # SQLite database
├── models/
│   ├── lstm_model.py        # LSTM for temporal patterns
│   ├── xgboost_model.py     # XGBoost for features
│   └── hybrid_model.py      # Ensemble model
├── trading/
│   ├── engine/              # Paper trading + portfolio manager
│   ├── strategies/          # ML swing strategy
│   ├── risk/                # Risk management
│   └── orchestrator.py      # Multi-coin coordinator
├── analysis/                # Performance metrics
├── scripts/                 # Entry point scripts
├── Dockerfile               # Container definition
└── docker-compose.yml       # Container orchestration
```

## Configuration

### Trading Parameters (`config/trading_config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| initial_capital | $10,000 | Starting paper money |
| max_position_pct | 20% | Max size per position |
| max_total_exposure | 80% | Max total invested |
| max_drawdown_pct | 15% | Circuit breaker |
| risk_per_trade_pct | 2% | Risk amount per trade |
| stop_loss_atr_multiple | 2.0 | Stop-loss in ATR units |
| take_profit_atr_multiple | 3.0 | Take-profit in ATR units |

### Category Allocation (`config/coin_registry.py`)

Each category has:
- **max_allocation_pct**: Maximum % of portfolio
- **max_coins**: Maximum positions in category
- **position_size_multiplier**: Adjust sizing (e.g., 0.5 for memecoins)

## Docker Commands

```bash
# Build image
docker-compose build

# Download data for all coins
docker-compose --profile data run data-fetcher

# Train all models
docker-compose --profile training run model-trainer

# Run backtest
docker-compose --profile backtest run backtester

# Start live paper trading
docker-compose up -d crypto-trader

# View logs
docker-compose logs -f crypto-trader

# Stop all services
docker-compose down
```

## Example Output

```
MULTI-COIN PORTFOLIO SUMMARY
======================================================================
Total Value:      $12,543.21
Cash Balance:     $4,231.54
Positions Value:  $8,311.67
Num Positions:    7

--- Category Allocations ---
  large_cap    | Current: 25.3% | Target: 40.0% | Value: $3,172.43 | Positions: 2
  altcoin      | Current: 18.7% | Target: 30.0% | Value: $2,345.12 | Positions: 3
  defi         | Current:  8.2% | Target: 15.0% | Value: $1,028.34 | Positions: 1
  memecoin     | Current:  5.1% | Target: 10.0% | Value:   $639.45 | Positions: 1
  ai           | Current:  4.5% | Target: 10.0% | Value:   $564.22 | Positions: 1
  layer2       | Current:  4.4% | Target: 15.0% | Value:   $552.11 | Positions: 1

--- Open Positions ---
  BTC      | Qty: 0.023451 | Entry: $67,234.12 | Current: $68,432.11 | P&L: +$28.12 (+1.8%) | [large_cap]
  ETH      | Qty: 0.412300 | Entry:  $3,423.45 | Current:  $3,512.34 | P&L: +$36.65 (+2.6%) | [large_cap]
  SOL      | Qty: 8.234500 | Entry:    $142.34 | Current:    $148.23 | P&L: +$48.52 (+4.1%) | [altcoin]
  ...
======================================================================
```

## Performance Targets

- **Directional Accuracy**: >55%
- **Sharpe Ratio**: >1.0
- **Max Drawdown**: <15%
- **Win Rate**: >50%

## Risk Disclaimers

- This is for **educational and paper trading purposes only**
- Past performance does not guarantee future results
- Cryptocurrency trading involves significant risk
- Paper trade for at least 30 days before considering real money
- Never invest more than you can afford to lose
- Memecoins are extremely high risk - allocation is limited to 10%

## License

MIT
