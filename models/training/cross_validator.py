"""Time series cross-validation for trading models."""
import pandas as pd
from typing import Generator, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CVSplit:
    """Cross-validation split information."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int


class WalkForwardValidator:
    """
    Walk-forward validation for time series trading models.

    Prevents look-ahead bias by always training on past data
    and validating on future data.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 365,
        test_size: int = 30,
        gap: int = 0
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of train/test splits
            train_size: Number of days in training window
            test_size: Number of days in test window
            gap: Gap between train and test (for prediction horizon)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, CVSplit], None, None]:
        """
        Generate train/test splits using walk-forward methodology.

        Args:
            df: DataFrame with datetime index

        Yields:
            Tuples of (train_df, test_df, split_info)
        """
        n_samples = len(df)
        logger.info(f"Walk-forward CV: {n_samples} samples, {self.n_splits} splits")

        for i in range(self.n_splits):
            # Calculate split indices (from end backwards)
            test_end = n_samples - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.gap
            train_start = max(0, train_end - self.train_size)

            if train_start >= train_end or test_start >= test_end:
                logger.warning(f"Skipping fold {i+1}: insufficient data")
                continue

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            split_info = CVSplit(
                fold=i + 1,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_size=len(train_df),
                test_size=len(test_df)
            )

            logger.info(
                f"Fold {i+1}: Train {split_info.train_start.date()} to {split_info.train_end.date()} "
                f"({split_info.train_size} samples), "
                f"Test {split_info.test_start.date()} to {split_info.test_end.date()} "
                f"({split_info.test_size} samples)"
            )

            yield train_df, test_df, split_info

    def expanding_window_split(
        self,
        df: pd.DataFrame,
        initial_train_size: int = 365
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, CVSplit], None, None]:
        """
        Expanding window: training set grows with each split.

        More suitable for strategies that benefit from more data.

        Args:
            df: DataFrame with datetime index
            initial_train_size: Minimum training set size

        Yields:
            Tuples of (train_df, test_df, split_info)
        """
        n_samples = len(df)

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.gap
            train_start = 0  # Always start from beginning

            if train_end - train_start < initial_train_size:
                logger.warning(f"Skipping fold {i+1}: train size below minimum")
                continue

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            split_info = CVSplit(
                fold=i + 1,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                train_size=len(train_df),
                test_size=len(test_df)
            )

            logger.info(
                f"Fold {i+1} (expanding): Train {split_info.train_size} samples, "
                f"Test {split_info.test_size} samples"
            )

            yield train_df, test_df, split_info
