"""Feature engineering for swing trading ML models."""
import numpy as np
import pandas as pd
import ta
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

        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        # Simple Moving Averages - Multiple timeframes
        for period in [7, 14, 21, 50, 200]:
            self.df[f"sma_{period}"] = ta.trend.sma_indicator(close, window=period)

        # Exponential Moving Averages
        for period in [7, 14, 21, 50, 200]:
            self.df[f"ema_{period}"] = ta.trend.ema_indicator(close, window=period)

        # MACD - Primary trend indicator
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self.df["MACD_12_26_9"] = macd.macd()
        self.df["MACDh_12_26_9"] = macd.macd_diff()
        self.df["MACDs_12_26_9"] = macd.macd_signal()

        # ADX - Average Directional Index (trend strength)
        adx = ta.trend.ADXIndicator(high, low, close, window=14)
        self.df["ADX_14"] = adx.adx()
        self.df["DMP_14"] = adx.adx_pos()
        self.df["DMN_14"] = adx.adx_neg()

        # Price relative to moving averages
        if "sma_50" in self.df.columns:
            self.df["close_to_sma_50"] = close / self.df["sma_50"]
        if "sma_200" in self.df.columns:
            self.df["close_to_sma_200"] = close / self.df["sma_200"]
        if "sma_50" in self.df.columns and "sma_200" in self.df.columns:
            self.df["sma_50_to_200"] = self.df["sma_50"] / self.df["sma_200"]

        # Moving average crossovers
        if "ema_7" in self.df.columns and "ema_21" in self.df.columns:
            self.df["ema_7_above_21"] = (self.df["ema_7"] > self.df["ema_21"]).astype(int)

    def _add_momentum_indicators(self):
        """Add momentum oscillators."""
        logger.debug("Adding momentum indicators...")

        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        # RSI - Multiple timeframes
        for period in [7, 14, 21]:
            self.df[f"rsi_{period}"] = ta.momentum.rsi(close, window=period)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        self.df["STOCHk_14_3_3"] = stoch.stoch()
        self.df["STOCHd_14_3_3"] = stoch.stoch_signal()

        # Williams %R
        self.df["willr"] = ta.momentum.williams_r(high, low, close)

        # CCI - Commodity Channel Index
        self.df["cci"] = ta.trend.cci(high, low, close)

        # Rate of Change
        for period in [7, 14]:
            self.df[f"roc_{period}"] = ta.momentum.roc(close, window=period)

        # Ultimate Oscillator
        self.df["uo"] = ta.momentum.ultimate_oscillator(high, low, close)

        # TSI - True Strength Index
        self.df["tsi"] = ta.momentum.tsi(close)

    def _add_volatility_indicators(self):
        """Add volatility measures (critical for position sizing)."""
        logger.debug("Adding volatility indicators...")

        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        self.df["BBL_20_2.0"] = bb.bollinger_lband()
        self.df["BBM_20_2.0"] = bb.bollinger_mavg()
        self.df["BBU_20_2.0"] = bb.bollinger_hband()
        self.df["BBB_20_2.0"] = bb.bollinger_wband()
        self.df["BBP_20_2.0"] = bb.bollinger_pband()

        # ATR - Average True Range (for stop-loss calculation)
        for period in [14, 21]:
            self.df[f"atr_{period}"] = ta.volatility.average_true_range(high, low, close, window=period)

        # ATR as percentage of price
        if "atr_14" in self.df.columns:
            self.df["atr_pct"] = self.df["atr_14"] / close * 100

        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(high, low, close)
        self.df["KCLe_20_2"] = kc.keltner_channel_lband()
        self.df["KCBe_20_2"] = kc.keltner_channel_mband()
        self.df["KCUe_20_2"] = kc.keltner_channel_hband()

        # Historical Volatility (annualized)
        returns = close.pct_change()
        self.df["hvol_14"] = returns.rolling(14).std() * np.sqrt(365)
        self.df["hvol_30"] = returns.rolling(30).std() * np.sqrt(365)

        # Donchian Channels (breakout detection)
        dc = ta.volatility.DonchianChannel(high, low, close, window=20)
        self.df["DCL_20_20"] = dc.donchian_channel_lband()
        self.df["DCM_20_20"] = dc.donchian_channel_mband()
        self.df["DCU_20_20"] = dc.donchian_channel_hband()

        # Ulcer Index
        self.df["ulcer_index"] = ta.volatility.ulcer_index(close)

    def _add_volume_indicators(self):
        """Add volume-based indicators."""
        logger.debug("Adding volume indicators...")

        if self.df["volume"].isna().all() or (self.df["volume"] == 0).all():
            logger.warning("Volume data not available, skipping volume indicators")
            return

        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        volume = self.df["volume"]

        # OBV - On-Balance Volume
        self.df["obv"] = ta.volume.on_balance_volume(close, volume)

        # Volume SMA
        self.df["volume_sma_20"] = ta.trend.sma_indicator(volume, window=20)

        # Volume ratio
        if "volume_sma_20" in self.df.columns:
            self.df["volume_ratio"] = volume / self.df["volume_sma_20"]

        # MFI - Money Flow Index
        self.df["mfi"] = ta.volume.money_flow_index(high, low, close, volume)

        # CMF - Chaikin Money Flow
        self.df["cmf"] = ta.volume.chaikin_money_flow(high, low, close, volume)

        # AD - Accumulation/Distribution
        self.df["ad"] = ta.volume.acc_dist_index(high, low, close, volume)

        # Force Index
        self.df["force_index"] = ta.volume.force_index(close, volume)

        # VWAP (approximation for daily)
        self.df["vwap"] = ta.volume.volume_weighted_average_price(high, low, close, volume)

    def _add_price_patterns(self):
        """Add price pattern features."""
        logger.debug("Adding price patterns...")

        close = self.df["close"]

        # Returns at various horizons
        for period in [1, 3, 7, 14, 21]:
            self.df[f"return_{period}d"] = close.pct_change(period)

        # Log returns (more normally distributed)
        self.df["log_return_1d"] = np.log(close / close.shift(1))

        # High-Low spread (daily range)
        self.df["hl_spread"] = (self.df["high"] - self.df["low"]) / close

        # Close position within day's range
        hl_range = self.df["high"] - self.df["low"]
        self.df["close_position"] = np.where(
            hl_range > 0,
            (close - self.df["low"]) / hl_range,
            0.5
        )

        # Gap detection
        self.df["gap_up"] = (self.df["open"] > close.shift(1)).astype(int)
        self.df["gap_down"] = (self.df["open"] < close.shift(1)).astype(int)
        self.df["gap_pct"] = (self.df["open"] - close.shift(1)) / close.shift(1)

        # Consecutive up/down days
        self.df["up_day"] = (close > close.shift(1)).astype(int)

        # Rolling highs/lows
        self.df["rolling_high_20"] = self.df["high"].rolling(20).max()
        self.df["rolling_low_20"] = self.df["low"].rolling(20).min()

        # Distance from 20-day high/low
        self.df["dist_from_high_20"] = (close - self.df["rolling_high_20"]) / self.df["rolling_high_20"]
        self.df["dist_from_low_20"] = (close - self.df["rolling_low_20"]) / self.df["rolling_low_20"]

        # Price momentum (close vs close n days ago)
        for period in [5, 10, 20]:
            self.df[f"price_momentum_{period}"] = close / close.shift(period) - 1

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

        close = self.df["close"]

        # Regression targets: future returns
        for horizon in [1, 3, 7, 14]:
            self.df[f"target_return_{horizon}d"] = (
                close.shift(-horizon) / close - 1
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
