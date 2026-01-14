#!/usr/bin/env python3
"""Train the hybrid ML model."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.database import TradingDatabase
from data.processors.feature_engineer import FeatureEngineer
from models.hybrid_model import HybridModel
from config.model_config import ModelConfig
from utils.logger import setup_logger

logger = setup_logger("train_model")


def main():
    """Train hybrid model on historical data."""
    # Initialize
    db = TradingDatabase("data/trading.db")
    config = ModelConfig()

    # Load data
    symbol = "bitcoin"  # Primary training symbol
    logger.info(f"Loading data for {symbol}...")
    df = db.load_ohlcv(symbol)

    if len(df) < 100:
        logger.error(f"Insufficient data: only {len(df)} rows. Need at least 100.")
        logger.info("Run 'python scripts/download_data.py' first to fetch data.")
        return

    logger.info(f"Loaded {len(df)} rows of data")

    # Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(df)
    df_features = engineer.add_all_features(prediction_horizon=config.prediction_horizon)

    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    logger.info(f"Generated {len(feature_cols)} features")

    # Clean data
    df_clean = engineer.get_clean_features(dropna=True)
    logger.info(f"Clean dataset: {len(df_clean)} rows")

    if len(df_clean) < 100:
        logger.error("Insufficient clean data after dropping NaN rows")
        return

    # Train model
    logger.info("Training hybrid model...")
    model = HybridModel(config)

    results = model.train(
        df_clean,
        feature_cols=feature_cols,
        target_col=f"target_direction_{config.prediction_horizon}d",
        validation_split=config.validation_split
    )

    # Print results
    logger.info("Training Results:")
    if "lstm_val_metrics" in results:
        lstm_metrics = results["lstm_val_metrics"]
        logger.info(f"  LSTM - Accuracy: {lstm_metrics.get('accuracy', 0):.4f}, "
                   f"AUC: {lstm_metrics.get('auc_roc', 0):.4f}")

    if "xgb_val_metrics" in results:
        xgb_metrics = results["xgb_val_metrics"]
        logger.info(f"  XGBoost - Accuracy: {xgb_metrics.get('accuracy', 0):.4f}, "
                   f"AUC: {xgb_metrics.get('auc_roc', 0):.4f}")

    # Print top features
    if "xgb" in results and "feature_importance" in results["xgb"]:
        logger.info("\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(results["xgb"]["feature_importance"].items())[:10]):
            logger.info(f"  {i+1}. {feat}: {imp:.4f}")

    # Save model
    save_path = Path("models/saved/hybrid_model")
    model.save(str(save_path))
    logger.info(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
