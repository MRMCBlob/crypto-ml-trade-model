"""Feature engineering for swing trading ML models."""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for swing trading.

    Generates 50+ technical indicators optimized for swing trading
    (positions held days to weeks).
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime
        """
        self.df = df.copy()
        self._validate_data()

    def _validate_data(self):
        """Validate input data has required columns."""
        required = ["open", "high", "low", "close"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "volume" not in self.df.columns:
            logger.warning("Volume column not found, setting to 0")
            self.df["volume"] = 0

    def add_all_features(self, prediction_horizon: int = 7) -> pd.DataFrame:
        """
        Add all swing trading features.

        Args:
            prediction_horizon: Days ahead for target variables

        Returns:
            DataFrame with all features added
        """
        logger.info("Starting feature engineering...")

        self._add_trend_indicators()
        self._add_momentum_indicators()
        self._add_volatility_indicators()
        self._add_volume_indicators()
        self._add_price_patterns()
        self._add_temporal_features()
        self._add_target_variables(prediction_horizon)

        feature_count = len([c for c in self.df.columns if c not in
                            ["open", "high", "low", "close", "volume"]])
        logger.info(f"Feature engineering complete. Added {feature_count} features.")

        return self.df

    def _add_trend_indicators(self):
        """Add trend-following indicators (crucial for swing trading)."""
        logger.debug("Adding trend indicators...")

        # Simple Moving Averages - Multiple timeframes
        for period in [7, 14, 21, 50, 200]:
            self.df[f"sma_{period}"] = ta.sma(self.df["close"], length=period)

        # Exponential Moving Averages
        for period in [7, 14, 21, 50, 200]:
            self.df[f"ema_{period}"] = ta.ema(self.df["close"], length=period)

        # MACD - Primary trend indicator
        macd = ta.macd(self.df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            self.df = pd.concat([self.df, macd], axis=1)

        # ADX - Average Directional Index (trend strength)
        adx = ta.adx(self.df["high"], self.df["low"], self.df["close"], length=14)
        if adx is not None:
            self.df = pd.concat([self.df, adx], axis=1)

        # Supertrend (good for swing entries)
        supertrend = ta.supertrend(
            self.df["high"], self.df["low"], self.df["close"],
            length=10, multiplier=3.0
        )
        if supertrend is not None:
            self.df = pd.concat([self.df, supertrend], axis=1)

        # Price relative to moving averages
        if "sma_50" in self.df.columns:
            self.df["close_to_sma_50"] = self.df["close"] / self.df["sma_50"]
        if "sma_200" in self.df.columns:
            self.df["close_to_sma_200"] = self.df["close"] / self.df["sma_200"]
        if "sma_50" in self.df.columns and "sma_200" in self.df.columns:
            self.df["sma_50_to_200"] = self.df["sma_50"] / self.df["sma_200"]

        # Moving average crossovers
        if "ema_7" in self.df.columns and "ema_21" in self.df.columns:
            self.df["ema_7_above_21"] = (self.df["ema_7"] > self.df["ema_21"]).astype(int)

    def _add_momentum_indicators(self):
        """Add momentum oscillators."""
        logger.debug("Adding momentum indicators...")

        # RSI - Multiple timeframes
        for period in [7, 14, 21]:
            rsi = ta.rsi(self.df["close"], length=period)
            if rsi is not None:
                self.df[f"rsi_{period}"] = rsi

        # Stochastic Oscillator
        stoch = ta.stoch(self.df["high"], self.df["low"], self.df["close"])
        if stoch is not None:
            self.df = pd.concat([self.df, stoch], axis=1)

        # Williams %R
        willr = ta.willr(self.df["high"], self.df["low"], self.df["close"])
        if willr is not None:
            self.df["willr"] = willr

        # CCI - Commodity Channel Index
        cci = ta.cci(self.df["high"], self.df["low"], self.df["close"])
        if cci is not None:
            self.df["cci"] = cci

        # Rate of Change
        for period in [7, 14]:
            roc = ta.roc(self.df["close"], length=period)
            if roc is not None:
                self.df[f"roc_{period}"] = roc

        # Momentum
        for period in [7, 14]:
            mom = ta.mom(self.df["close"], length=period)
            if mom is not None:
                self.df[f"mom_{period}"] = mom

        # Ultimate Oscillator
        uo = ta.uo(self.df["high"], self.df["low"], self.df["close"])
        if uo is not None:
            self.df["uo"] = uo

    def _add_volatility_indicators(self):
        """Add volatility measures (critical for position sizing)."""
        logger.debug("Adding volatility indicators...")

        # Bollinger Bands
        bbands = ta.bbands(self.df["close"], length=20, std=2.0)
        if bbands is not None:
            self.df = pd.concat([self.df, bbands], axis=1)

        # ATR - Average True Range (for stop-loss calculation)
        for period in [14, 21]:
            atr = ta.atr(self.df["high"], self.df["low"], self.df["close"], length=period)
            if atr is not None:
                self.df[f"atr_{period}"] = atr

        # ATR as percentage of price
        if "atr_14" in self.df.columns:
            self.df["atr_pct"] = self.df["atr_14"] / self.df["close"] * 100

        # Keltner Channels
        kc = ta.kc(self.df["high"], self.df["low"], self.df["close"])
        if kc is not None:
            self.df = pd.concat([self.df, kc], axis=1)

        # Historical Volatility (annualized)
        returns = self.df["close"].pct_change()
        self.df["hvol_14"] = returns.rolling(14).std() * np.sqrt(365)
        self.df["hvol_30"] = returns.rolling(30).std() * np.sqrt(365)

        # Donchian Channels (breakout detection)
        donchian = ta.donchian(self.df["high"], self.df["low"],
                               lower_length=20, upper_length=20)
        if donchian is not None:
            self.df = pd.concat([self.df, donchian], axis=1)

        # True Range
        tr = ta.true_range(self.df["high"], self.df["low"], self.df["close"])
        if tr is not None:
            self.df["true_range"] = tr

    def _add_volume_indicators(self):
        """Add volume-based indicators."""
        logger.debug("Adding volume indicators...")

        if self.df["volume"].isna().all() or (self.df["volume"] == 0).all():
            logger.warning("Volume data not available, skipping volume indicators")
            return

        # OBV - On-Balance Volume
        obv = ta.obv(self.df["close"], self.df["volume"])
        if obv is not None:
            self.df["obv"] = obv

        # Volume SMA
        self.df["volume_sma_20"] = ta.sma(self.df["volume"], length=20)

        # Volume ratio
        if "volume_sma_20" in self.df.columns:
            self.df["volume_ratio"] = self.df["volume"] / self.df["volume_sma_20"]

        # MFI - Money Flow Index
        mfi = ta.mfi(self.df["high"], self.df["low"], self.df["close"], self.df["volume"])
        if mfi is not None:
            self.df["mfi"] = mfi

        # CMF - Chaikin Money Flow
        cmf = ta.cmf(self.df["high"], self.df["low"], self.df["close"], self.df["volume"])
        if cmf is not None:
            self.df["cmf"] = cmf

        # AD - Accumulation/Distribution
        ad = ta.ad(self.df["high"], self.df["low"], self.df["close"], self.df["volume"])
        if ad is not None:
            self.df["ad"] = ad

        # VWAP (approximation for daily)
        vwap = ta.vwap(self.df["high"], self.df["low"], self.df["close"], self.df["volume"])
        if vwap is not None:
            self.df["vwap"] = vwap

    def _add_price_patterns(self):
        """Add price pattern features."""
        logger.debug("Adding price patterns...")

        # Returns at various horizons
        for period in [1, 3, 7, 14, 21]:
            self.df[f"return_{period}d"] = self.df["close"].pct_change(period)

        # Log returns (more normally distributed)
        self.df["log_return_1d"] = np.log(self.df["close"] / self.df["close"].shift(1))

        # High-Low spread (daily range)
        self.df["hl_spread"] = (self.df["high"] - self.df["low"]) / self.df["close"]

        # Close position within day's range
        hl_range = self.df["high"] - self.df["low"]
        self.df["close_position"] = np.where(
            hl_range > 0,
            (self.df["close"] - self.df["low"]) / hl_range,
            0.5
        )

        # Gap detection
        self.df["gap_up"] = (self.df["open"] > self.df["close"].shift(1)).astype(int)
        self.df["gap_down"] = (self.df["open"] < self.df["close"].shift(1)).astype(int)
        self.df["gap_pct"] = (self.df["open"] - self.df["close"].shift(1)) / self.df["close"].shift(1)

        # Consecutive up/down days
        self.df["up_day"] = (self.df["close"] > self.df["close"].shift(1)).astype(int)

        # Rolling highs/lows
        self.df["rolling_high_20"] = self.df["high"].rolling(20).max()
        self.df["rolling_low_20"] = self.df["low"].rolling(20).min()

        # Distance from 20-day high/low
        self.df["dist_from_high_20"] = (self.df["close"] - self.df["rolling_high_20"]) / self.df["rolling_high_20"]
        self.df["dist_from_low_20"] = (self.df["close"] - self.df["rolling_low_20"]) / self.df["rolling_low_20"]

        # Price momentum (close vs close n days ago)
        for period in [5, 10, 20]:
            self.df[f"price_momentum_{period}"] = self.df["close"] / self.df["close"].shift(period) - 1

    def _add_temporal_features(self):
        """Add time-based features."""
        logger.debug("Adding temporal features...")

        if not isinstance(self.df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping temporal features")
            return

        # Day of week (crypto trades 24/7, but sentiment varies)
        self.df["day_of_week"] = self.df.index.dayofweek
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)

        # Month (seasonality)
        self.df["month"] = self.df.index.month

        # Quarter
        self.df["quarter"] = self.df.index.quarter

        # Day of month
        self.df["day_of_month"] = self.df.index.day

        # Week of year
        self.df["week_of_year"] = self.df.index.isocalendar().week.astype(int)

    def _add_target_variables(self, prediction_horizon: int = 7):
        """
        Add prediction targets for ML training.

        Args:
            prediction_horizon: Primary prediction horizon in days
        """
        logger.debug(f"Adding target variables (horizon={prediction_horizon}d)...")

        # Regression targets: future returns
        for horizon in [1, 3, 7, 14]:
            self.df[f"target_return_{horizon}d"] = (
                self.df["close"].shift(-horizon) / self.df["close"] - 1
            )

        # Classification targets: direction (up=1, down=0)
        for horizon in [1, 3, 7, 14]:
            self.df[f"target_direction_{horizon}d"] = (
                self.df[f"target_return_{horizon}d"] > 0
            ).astype(int)

        # Swing trading specific: significant moves
        # 1 = up >3%, 0 = neutral, -1 = down >3%
        threshold = 0.03  # 3% move threshold
        target_col = f"target_return_{prediction_horizon}d"
        if target_col in self.df.columns:
            self.df["target_swing"] = np.where(
                self.df[target_col] > threshold, 1,
                np.where(self.df[target_col] < -threshold, -1, 0)
            )

        # Binary swing target: significant up move
        self.df["target_swing_up"] = (self.df[target_col] > threshold).astype(int)

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names (excluding targets and OHLCV)."""
        exclude_prefixes = ["target_", "open", "high", "low", "close", "volume"]
        feature_cols = [
            col for col in self.df.columns
            if not any(col.startswith(prefix) or col == prefix for prefix in exclude_prefixes)
        ]
        return feature_cols

    def get_clean_features(self, dropna: bool = True) -> pd.DataFrame:
        """
        Get feature DataFrame with NaN handling.

        Args:
            dropna: Whether to drop rows with NaN values

        Returns:
            Clean DataFrame ready for ML
        """
        df = self.df.copy()

        if dropna:
            # Drop rows with NaN in any column
            initial_len = len(df)
            df = df.dropna()
            dropped = initial_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NaN values")

        return df

    def get_feature_target_split(
        self,
        target_col: str = "target_direction_7d"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.

        Args:
            target_col: Name of target column

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        df = self.get_clean_features(dropna=True)
        feature_cols = self.get_feature_columns()

        X = df[feature_cols]
        y = df[target_col]

        return X, y
