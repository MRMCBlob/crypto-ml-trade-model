"""Abstract base class for ML models."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary with training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Input features

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions.

        Args:
            X: Input features

        Returns:
            Probability predictions (for classification)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
        }

        if len(np.unique(y)) > 1:
            metrics["auc_roc"] = roc_auc_score(y, y_proba)

        return metrics
