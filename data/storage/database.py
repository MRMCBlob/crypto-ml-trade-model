"""SQLite database manager for trading data."""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class TradingDatabase:
    """SQLite database manager for trading data storage."""

    def __init__(self, db_path: str = "storage/trading.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self.connection() as conn:
            conn.executescript("""
                -- OHLCV price data
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL DEFAULT 'daily',
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
                ON ohlcv(symbol, timeframe, timestamp);

                -- Trading orders
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    status TEXT NOT NULL,
                    filled_price REAL,
                    filled_timestamp DATETIME,
                    fees REAL DEFAULT 0,
                    strategy TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Executed trades
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    timestamp DATETIME NOT NULL,
                    exit_timestamp DATETIME,
                    pnl REAL,
                    fees REAL DEFAULT 0,
                    strategy TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Portfolio snapshots
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions_json TEXT,
                    daily_return REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp
                ON portfolio_snapshots(timestamp);

                -- Model predictions log
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    model_name TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL,
                    signal TEXT,
                    actual_return REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Feature data cache
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                );
            """)

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str = "daily"):
        """
        Save OHLCV data to database.

        Args:
            df: DataFrame with OHLCV data (index=datetime)
            symbol: Cryptocurrency symbol
            timeframe: Data timeframe (default: daily)
        """
        df_copy = df.copy()
        df_copy["symbol"] = symbol
        df_copy["timeframe"] = timeframe
        df_copy = df_copy.reset_index()
        df_copy = df_copy.rename(columns={"index": "timestamp"})

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close"]
        for col in required:
            if col not in df_copy.columns:
                raise ValueError(f"Missing required column: {col}")

        if "volume" not in df_copy.columns:
            df_copy["volume"] = 0

        with self.connection() as conn:
            for _, row in df_copy.iterrows():
                # Convert pandas Timestamp to ISO format string for SQLite
                timestamp = row["timestamp"]
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()

                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["symbol"], row["timeframe"], timestamp,
                    row["open"], row["high"], row["low"], row["close"], row["volume"]
                ))

        logger.info(f"Saved {len(df_copy)} OHLCV rows for {symbol}")

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Cryptocurrency symbol
            timeframe: Data timeframe
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        with self.connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)

        logger.info(f"Loaded {len(df)} OHLCV rows for {symbol}")
        return df

    def save_trade(self, trade: Dict[str, Any]):
        """Save a completed trade to database."""
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO trades
                (symbol, side, quantity, entry_price, exit_price, timestamp,
                 exit_timestamp, pnl, fees, strategy, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["symbol"], trade["side"], trade["quantity"],
                trade["entry_price"], trade.get("exit_price"),
                trade["timestamp"], trade.get("exit_timestamp"),
                trade.get("pnl"), trade.get("fees", 0),
                trade.get("strategy"), trade.get("notes")
            ))

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Get trades from database."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def save_portfolio_snapshot(
        self,
        timestamp: datetime,
        total_value: float,
        cash_balance: float,
        positions: Dict[str, Any],
        daily_return: float = 0.0
    ):
        """Save portfolio snapshot to database."""
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash_balance, positions_json, daily_return)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp, total_value, cash_balance,
                json.dumps(positions), daily_return
            ))

    def get_portfolio_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get portfolio value history."""
        query = "SELECT timestamp, total_value, cash_balance, daily_return FROM portfolio_snapshots WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        with self.connection() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)

        return df

    def save_prediction(
        self,
        symbol: str,
        timestamp: datetime,
        model_name: str,
        prediction: float,
        confidence: Optional[float] = None,
        signal: Optional[str] = None
    ):
        """Log a model prediction."""
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO predictions
                (symbol, timestamp, model_name, prediction, confidence, signal)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, timestamp, model_name, prediction, confidence, signal))

    def get_symbol_count(self) -> Dict[str, int]:
        """Get count of records per symbol."""
        with self.connection() as conn:
            cursor = conn.execute("""
                SELECT symbol, COUNT(*) as count
                FROM ohlcv
                GROUP BY symbol
            """)
            return {row["symbol"]: row["count"] for row in cursor.fetchall()}
