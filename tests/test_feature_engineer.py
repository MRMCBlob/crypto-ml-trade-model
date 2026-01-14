"""Tests for feature engineering module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processors.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, 200)
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
        "high": prices * (1 + np.random.uniform(0, 0.02, 200)),
        "low": prices * (1 - np.random.uniform(0, 0.02, 200)),
        "close": prices,
        "volume": np.random.uniform(1e9, 5e9, 200)
    }, index=dates)

    return df


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""

    def test_initialization(self, sample_ohlcv):
        """Test engineer initializes correctly."""
        engineer = FeatureEngineer(sample_ohlcv)
        assert len(engineer.df) == 200
        assert "close" in engineer.df.columns

    def test_missing_columns_raises(self):
        """Test error raised when required columns missing."""
        df = pd.DataFrame({"price": [100, 101, 102]})
        with pytest.raises(ValueError, match="Missing required columns"):
            FeatureEngineer(df)

    def test_add_all_features(self, sample_ohlcv):
        """Test adding all features."""
        engineer = FeatureEngineer(sample_ohlcv)
        df = engineer.add_all_features(prediction_horizon=7)

        # Should have many new columns
        assert len(df.columns) > 20

        # Check some expected features exist
        assert "sma_14" in df.columns
        assert "rsi_14" in df.columns
        assert "atr_14" in df.columns
        assert "target_direction_7d" in df.columns

    def test_feature_columns(self, sample_ohlcv):
        """Test getting feature column names."""
        engineer = FeatureEngineer(sample_ohlcv)
        engineer.add_all_features()

        feature_cols = engineer.get_feature_columns()

        # Should not include OHLCV or targets
        assert "open" not in feature_cols
        assert "close" not in feature_cols
        assert "target_direction_7d" not in feature_cols

        # Should include technical indicators
        assert any("sma" in col for col in feature_cols)
        assert any("rsi" in col for col in feature_cols)

    def test_clean_features(self, sample_ohlcv):
        """Test getting clean features without NaN."""
        engineer = FeatureEngineer(sample_ohlcv)
        engineer.add_all_features()

        df_clean = engineer.get_clean_features(dropna=True)

        # Should have no NaN values
        assert df_clean.isna().sum().sum() == 0

        # Should have fewer rows than original (due to indicator warmup)
        assert len(df_clean) < len(sample_ohlcv)

    def test_target_variables(self, sample_ohlcv):
        """Test target variable generation."""
        engineer = FeatureEngineer(sample_ohlcv)
        engineer.add_all_features(prediction_horizon=7)

        # Check target columns exist
        assert "target_return_7d" in engineer.df.columns
        assert "target_direction_7d" in engineer.df.columns
        assert "target_swing" in engineer.df.columns

        # Direction should be binary (0 or 1)
        direction = engineer.df["target_direction_7d"].dropna()
        assert set(direction.unique()).issubset({0, 1})

    def test_feature_target_split(self, sample_ohlcv):
        """Test splitting features and target."""
        engineer = FeatureEngineer(sample_ohlcv)
        engineer.add_all_features()

        X, y = engineer.get_feature_target_split(target_col="target_direction_7d")

        # X and y should have same length
        assert len(X) == len(y)

        # X should not contain target columns
        assert "target_direction_7d" not in X.columns

        # y should be the target
        assert y.name == "target_direction_7d"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
