#!/usr/bin/env python3
"""Train models for multiple coins."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.coin_registry import get_coin_registry, CoinCategory
from config.model_config import ModelConfig
from config.settings import get_settings
from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from utils.logger import setup_logger

logger = setup_logger("train_multi_coin", log_dir=Path("logs"))


def train_model_for_coin(
    coin_id: str,
    db: TradingDatabase,
    config: ModelConfig,
    models_dir: Path
) -> bool:
    """
    Train a model for a single coin.

    Returns:
        True if successful
    """
    logger.info(f"Training model for {coin_id}...")

    # Load data
    df = db.load_ohlcv(coin_id)

    if len(df) < 100:
        logger.warning(f"  Insufficient data for {coin_id}: {len(df)} rows")
        return False

    logger.info(f"  Loaded {len(df)} rows of data")

    # Feature engineering
    try:
        engineer = FeatureEngineer(df)
        df_features = engineer.add_all_features(
            prediction_horizon=config.prediction_horizon
        )
        feature_cols = engineer.get_feature_columns()
        df_clean = engineer.get_clean_features(dropna=True)

        if len(df_clean) < 100:
            logger.warning(f"  Insufficient clean data for {coin_id}")
            return False

        logger.info(f"  Generated {len(feature_cols)} features, {len(df_clean)} clean samples")

    except Exception as e:
        logger.error(f"  Feature engineering failed for {coin_id}: {e}")
        return False

    # Train model
    try:
        model = HybridModel(config)
        results = model.train(
            df_clean,
            feature_cols=feature_cols,
            target_col=f"target_direction_{config.prediction_horizon}d",
            validation_split=config.validation_split
        )

        # Log metrics
        if "xgb_val_metrics" in results:
            metrics = results["xgb_val_metrics"]
            logger.info(
                f"  XGBoost - Accuracy: {metrics.get('accuracy', 0):.4f}, "
                f"AUC: {metrics.get('auc_roc', 0):.4f}"
            )

        if "lstm_val_metrics" in results:
            metrics = results["lstm_val_metrics"]
            logger.info(
                f"  LSTM - Accuracy: {metrics.get('accuracy', 0):.4f}, "
                f"AUC: {metrics.get('auc_roc', 0):.4f}"
            )

        # Save model
        save_path = models_dir / coin_id
        model.save(str(save_path))
        logger.info(f"  Model saved to {save_path}")

        return True

    except Exception as e:
        logger.error(f"  Training failed for {coin_id}: {e}")
        return False


def main():
    """Train models for all coins with sufficient data."""
    logger.info("=" * 60)
    logger.info("MULTI-COIN MODEL TRAINING")
    logger.info("=" * 60)

    # Initialize
    settings = get_settings()
    registry = get_coin_registry()
    db = TradingDatabase(settings.database_path)
    config = ModelConfig()
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get all tradeable coins
    coins = registry.get_all_tradeable_coins()
    logger.info(f"Found {len(coins)} tradeable coins")

    # Check which have enough data
    counts = db.get_symbol_count()
    trainable = [
        coin for coin in coins
        if counts.get(coin.id, 0) >= 100
    ]
    logger.info(f"Coins with sufficient data: {len(trainable)}")

    # Train models
    successful = 0
    failed = []

    for i, coin in enumerate(trainable, 1):
        logger.info(f"\n[{i}/{len(trainable)}] {coin.name} ({coin.symbol}) [{coin.category.value}]")
        logger.info("-" * 50)

        if train_model_for_coin(coin.id, db, config, models_dir):
            successful += 1
        else:
            failed.append(coin.symbol)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful: {successful}/{len(trainable)}")

    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")

    # List trained models by category
    logger.info("\nTrained models by category:")
    for category in CoinCategory:
        cat_coins = [c for c in trainable if c.category == category]
        cat_trained = [c.symbol for c in cat_coins if c.symbol not in failed]
        if cat_trained:
            logger.info(f"  {category.value}: {', '.join(cat_trained)}")


if __name__ == "__main__":
    main()
