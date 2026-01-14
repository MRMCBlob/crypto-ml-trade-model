"""LSTM model for temporal pattern recognition."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from config.model_config import ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM model for sequence-based price prediction.

    Uses bidirectional LSTM layers to capture temporal patterns
    in price sequences.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize LSTM model.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM architecture."""
        # Import TensorFlow here to avoid loading it until needed
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(self.config.lstm_units, return_sequences=True)),
            Dropout(self.config.lstm_dropout),
            Bidirectional(LSTM(self.config.lstm_units // 2, return_sequences=False)),
            Dropout(self.config.lstm_dropout),
            Dense(32, activation="relu"),
            Dropout(self.config.lstm_dropout / 2),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lstm_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        return model

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "target_direction_7d",
        price_cols: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM.

        Args:
            df: DataFrame with price and target data
            target_col: Name of target column
            price_cols: Columns to use for sequences

        Returns:
            Tuple of (X sequences, y targets)
        """
        if price_cols is None:
            price_cols = ["open", "high", "low", "close"]
            if "volume" in df.columns and not df["volume"].isna().all():
                price_cols.append("volume")

        # Extract and scale price data
        price_data = df[price_cols].values
        price_scaled = self.scaler.fit_transform(price_data)

        # Create sequences
        X, y = [], []
        seq_len = self.config.sequence_length

        for i in range(seq_len, len(df)):
            if target_col in df.columns and not pd.isna(df[target_col].iloc[i]):
                X.append(price_scaled[i - seq_len:i])
                y.append(df[target_col].iloc[i])

        return np.array(X), np.array(y)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train LSTM model."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        logger.info(f"Training LSTM model with {len(X_train)} samples...")

        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self._build_model(input_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_auc" if X_val is not None else "auc",
                patience=self.config.lstm_early_stopping_patience,
                restore_best_weights=True,
                mode="max"
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Training
        validation_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        logger.info("LSTM training complete")

        return {"history": history.history}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        proba = self.model.predict(X, verbose=0).flatten()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X, verbose=0).flatten()

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(path / "lstm_model.keras")
        joblib.dump(self.scaler, path / "lstm_scaler.pkl")
        joblib.dump(self.config, path / "lstm_config.pkl")
        logger.info(f"LSTM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load model from disk."""
        import tensorflow as tf

        path = Path(path)
        instance = cls()
        instance.model = tf.keras.models.load_model(path / "lstm_model.keras")
        instance.scaler = joblib.load(path / "lstm_scaler.pkl")
        instance.config = joblib.load(path / "lstm_config.pkl")
        instance.is_trained = True
        logger.info(f"LSTM model loaded from {path}")
        return instance
