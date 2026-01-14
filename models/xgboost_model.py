"""XGBoost model for feature-based prediction."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from config.model_config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for swing trading direction prediction.

    Uses gradient boosting on technical indicator features.
    Research shows 55-56% directional accuracy achievable.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize XGBoost model.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_direction_7d"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix for XGBoost.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column

        Returns:
            Tuple of (X features, y targets)
        """
        self.feature_names = feature_cols

        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values

        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y.astype(int)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info(f"Training XGBoost model with {len(X_train)} samples...")

        # Initialize model with config parameters
        self.model = xgb.XGBClassifier(**self.config.xgb_params)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )

        self.is_trained = True

        # Get feature importance
        importance = dict(zip(
            self.feature_names if self.feature_names else [f"f{i}" for i in range(X_train.shape[1])],
            self.model.feature_importances_
        ))

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        logger.info("XGBoost training complete")
        logger.info(f"Top 10 features: {list(importance.items())[:10]}")

        return {
            "feature_importance": importance,
            "best_iteration": self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        importance = dict(zip(
            self.feature_names if self.feature_names else [f"f{i}" for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        ))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path / "xgb_model.pkl")
        joblib.dump(self.scaler, path / "xgb_scaler.pkl")
        joblib.dump(self.feature_names, path / "xgb_features.pkl")
        joblib.dump(self.config, path / "xgb_config.pkl")
        logger.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """Load model from disk."""
        path = Path(path)
        instance = cls()
        instance.model = joblib.load(path / "xgb_model.pkl")
        instance.scaler = joblib.load(path / "xgb_scaler.pkl")
        instance.feature_names = joblib.load(path / "xgb_features.pkl")
        instance.config = joblib.load(path / "xgb_config.pkl")
        instance.is_trained = True
        logger.info(f"XGBoost model loaded from {path}")
        return instance
