"""Hybrid LSTM + XGBoost ensemble model."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib

from .lstm_model import LSTMModel
from .xgboost_model import XGBoostModel
from config.model_config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridModel:
    """
    Hybrid LSTM + XGBoost ensemble for swing trading prediction.

    Architecture:
    1. LSTM captures temporal dependencies in price sequences
    2. XGBoost predicts based on technical indicators
    3. Weighted ensemble combines both for final prediction

    Based on research showing hybrid models outperform standalone approaches.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize hybrid model.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.lstm_model = LSTMModel(self.config)
        self.xgb_model = XGBoostModel(self.config)
        self.is_trained = False

        # Ensemble weights (XGBoost weighted higher per research)
        self.lstm_weight = self.config.lstm_weight
        self.xgb_weight = self.config.xgb_weight

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_direction_7d",
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train both LSTM and XGBoost models.

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names for XGBoost
            target_col: Target column name
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training results from both models
        """
        results = {}

        # Time-series split: use last portion for validation
        split_idx = int(len(df) * (1 - validation_split))
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]

        logger.info(f"Training set: {len(df_train)} samples")
        logger.info(f"Validation set: {len(df_val)} samples")

        # 1. Train LSTM
        logger.info("=" * 50)
        logger.info("Training LSTM model...")
        logger.info("=" * 50)

        X_lstm_train, y_lstm_train = self.lstm_model.prepare_sequences(df_train, target_col)
        X_lstm_val, y_lstm_val = self.lstm_model.prepare_sequences(df_val, target_col)

        if len(X_lstm_train) > 0:
            lstm_results = self.lstm_model.train(
                X_lstm_train, y_lstm_train,
                X_lstm_val, y_lstm_val
            )
            results["lstm"] = lstm_results

            # Evaluate LSTM
            lstm_metrics = self.lstm_model.evaluate(X_lstm_val, y_lstm_val)
            results["lstm_val_metrics"] = lstm_metrics
            logger.info(f"LSTM validation metrics: {lstm_metrics}")
        else:
            logger.warning("Insufficient data for LSTM training")

        # 2. Train XGBoost
        logger.info("=" * 50)
        logger.info("Training XGBoost model...")
        logger.info("=" * 50)

        X_xgb_train, y_xgb_train = self.xgb_model.prepare_features(
            df_train, feature_cols, target_col
        )

        # Prepare validation data using the fitted scaler
        X_xgb_val = self.xgb_model.scaler.transform(
            df_val[feature_cols].dropna().values
        )
        y_xgb_val = df_val[target_col].dropna().values.astype(int)

        xgb_results = self.xgb_model.train(
            X_xgb_train, y_xgb_train,
            X_xgb_val, y_xgb_val
        )
        results["xgb"] = xgb_results

        # Evaluate XGBoost
        xgb_metrics = self.xgb_model.evaluate(X_xgb_val, y_xgb_val)
        results["xgb_val_metrics"] = xgb_metrics
        logger.info(f"XGBoost validation metrics: {xgb_metrics}")

        self.is_trained = True
        logger.info("=" * 50)
        logger.info("Hybrid model training complete!")
        logger.info("=" * 50)

        return results

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from both models and ensemble.

        Args:
            df: DataFrame with features
            feature_cols: Feature column names for XGBoost

        Returns:
            Dictionary with predictions from each model and ensemble
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        # LSTM predictions
        X_lstm, _ = self.lstm_model.prepare_sequences(df, target_col="target_direction_7d")
        lstm_prob = self.lstm_model.predict_proba(X_lstm) if len(X_lstm) > 0 else np.array([])

        # XGBoost predictions
        X_xgb = self.xgb_model.scaler.transform(df[feature_cols].dropna().values)
        xgb_prob = self.xgb_model.predict_proba(X_xgb)

        # Align lengths (LSTM has shorter output due to sequence requirement)
        if len(lstm_prob) > 0 and len(xgb_prob) > 0:
            min_len = min(len(lstm_prob), len(xgb_prob))
            lstm_prob = lstm_prob[-min_len:]
            xgb_prob = xgb_prob[-min_len:]

            # Ensemble prediction
            ensemble_prob = (
                self.lstm_weight * lstm_prob +
                self.xgb_weight * xgb_prob
            )
        else:
            # Fall back to XGBoost only
            ensemble_prob = xgb_prob
            logger.warning("Using XGBoost only (LSTM predictions unavailable)")

        # Generate signals: 1 (buy), 0 (hold), -1 (sell)
        signal = np.where(
            ensemble_prob > 0.6, 1,
            np.where(ensemble_prob < 0.4, -1, 0)
        )

        return {
            "lstm_prob": lstm_prob,
            "xgb_prob": xgb_prob,
            "ensemble_prob": ensemble_prob,
            "signal": signal,
            "prediction": (ensemble_prob > 0.5).astype(int)
        }

    def predict_single(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[float, int, str]:
        """
        Get prediction for most recent data point.

        Args:
            df: DataFrame with features
            feature_cols: Feature column names

        Returns:
            Tuple of (probability, signal, signal_name)
        """
        predictions = self.predict(df, feature_cols)

        prob = predictions["ensemble_prob"][-1]
        signal = predictions["signal"][-1]

        signal_names = {1: "BUY", 0: "HOLD", -1: "SELL"}
        signal_name = signal_names.get(signal, "HOLD")

        return prob, signal, signal_name

    def save(self, path: str):
        """Save both models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save individual models
        self.lstm_model.save(path / "lstm")
        self.xgb_model.save(path / "xgb")

        # Save ensemble config
        ensemble_config = {
            "lstm_weight": self.lstm_weight,
            "xgb_weight": self.xgb_weight
        }
        joblib.dump(ensemble_config, path / "ensemble_config.pkl")

        logger.info(f"Hybrid model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridModel":
        """Load both models from disk."""
        path = Path(path)
        instance = cls()

        instance.lstm_model = LSTMModel.load(path / "lstm")
        instance.xgb_model = XGBoostModel.load(path / "xgb")

        ensemble_config = joblib.load(path / "ensemble_config.pkl")
        instance.lstm_weight = ensemble_config["lstm_weight"]
        instance.xgb_weight = ensemble_config["xgb_weight"]

        instance.is_trained = True
        logger.info(f"Hybrid model loaded from {path}")
        return instance
