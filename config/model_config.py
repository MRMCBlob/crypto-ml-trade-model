"""ML model configuration."""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # Data parameters
    sequence_length: int = 30            # Days of history for LSTM
    prediction_horizon: int = 7          # Days ahead to predict
    train_test_split: float = 0.8        # 80% train, 20% test
    validation_split: float = 0.2        # 20% of train for validation

    # LSTM parameters
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_early_stopping_patience: int = 15

    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
    })

    # Ensemble weights
    lstm_weight: float = 0.4
    xgb_weight: float = 0.6

    # Cross-validation
    cv_n_splits: int = 5
    cv_train_size: int = 365             # 1 year training window
    cv_test_size: int = 30               # 1 month test window

    # Feature selection
    min_feature_importance: float = 0.01  # Drop features below this importance
