"""Performance metrics calculation for trading strategies."""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive performance metrics report."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_trade_return_pct: float
    best_trade: float
    worst_trade: float
    avg_holding_period: float  # days

    # Additional metrics
    volatility: float
    avg_daily_return: float


class PerformanceAnalyzer:
    """
    Calculate comprehensive trading performance metrics.

    Implements industry-standard metrics for evaluating trading strategies.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize analyzer.

        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year (365 for daily crypto)

        Returns:
            Annualized Sharpe ratio
        """
        if returns.std() == 0 or len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Sortino Ratio (penalizes downside volatility only).

        More appropriate for trading strategies than Sharpe.

        Args:
            returns: Series of periodic returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std

        return sortino

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> tuple[float, float, int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Calculate maximum drawdown and its duration.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Tuple of (max_dd_dollars, max_dd_pct, duration_days, peak_date, trough_date)
        """
        if len(equity_curve) == 0:
            return 0.0, 0.0, 0, None, None

        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve - rolling_max
        drawdown_pct = drawdown / rolling_max

        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()

        trough_idx = drawdown.idxmin()
        peak_idx = equity_curve[:trough_idx].idxmax() if trough_idx else None

        duration = 0
        if peak_idx and trough_idx:
            if hasattr(trough_idx - peak_idx, 'days'):
                duration = (trough_idx - peak_idx).days
            else:
                # Handle case where index is not datetime
                duration = int(equity_curve.index.get_loc(trough_idx) -
                              equity_curve.index.get_loc(peak_idx))

        return max_dd, max_dd_pct, duration, peak_idx, trough_idx

    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).

        Good for evaluating risk-adjusted performance over longer periods.

        Args:
            returns: Series of periodic returns
            equity_curve: Series of portfolio values
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        annualized_return = returns.mean() * periods_per_year
        _, max_dd_pct, _, _, _ = self.calculate_max_drawdown(equity_curve)

        if max_dd_pct == 0:
            return np.inf if annualized_return > 0 else 0.0

        return annualized_return / abs(max_dd_pct)

    def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trade statistics.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary of trade statistics
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "avg_trade_return_pct": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "avg_holding_period": 0.0,
                "total_pnl": 0.0
            }

        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t.get("pnl_pct", 0) for t in trades]
        holding_days = [t.get("holding_days", 0) for t in trades]

        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        profit_factor = sum(winning) / abs(sum(losing)) if losing else np.inf

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(trades),
            "profit_factor": profit_factor,
            "avg_trade_return": np.mean(pnls),
            "avg_trade_return_pct": np.mean(pnl_pcts),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_holding_period": np.mean(holding_days) if holding_days else 0,
            "total_pnl": sum(pnls)
        }

    def generate_report(
        self,
        equity_curve: pd.Series,
        trades: List[Dict[str, Any]],
        initial_capital: float
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            equity_curve: Series of portfolio values (datetime index)
            trades: List of completed trades
            initial_capital: Starting capital

        Returns:
            PerformanceReport with all metrics
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Total return
        final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        total_return = final_value - initial_capital
        total_return_pct = (final_value / initial_capital - 1) * 100

        # Annualized return
        if len(equity_curve) > 1:
            trading_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if trading_days > 0:
                annualized_return = ((1 + total_return_pct / 100) ** (365 / trading_days) - 1) * 100
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        # Risk metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, equity_curve)
        max_dd, max_dd_pct, max_dd_duration, _, _ = self.calculate_max_drawdown(equity_curve)

        # Trade analysis
        trade_stats = self.analyze_trades(trades)

        # Volatility
        volatility = returns.std() * np.sqrt(365) * 100 if len(returns) > 0 else 0.0

        return PerformanceReport(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct * 100,
            max_drawdown_duration=max_dd_duration,
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"] * 100,
            profit_factor=trade_stats["profit_factor"],
            avg_trade_return=trade_stats["avg_trade_return"],
            avg_trade_return_pct=trade_stats["avg_trade_return_pct"],
            best_trade=trade_stats["best_trade"],
            worst_trade=trade_stats["worst_trade"],
            avg_holding_period=trade_stats["avg_holding_period"],
            volatility=volatility,
            avg_daily_return=returns.mean() * 100 if len(returns) > 0 else 0.0
        )

    def print_report(self, report: PerformanceReport):
        """Print formatted performance report."""
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)

        print("\n--- Returns ---")
        print(f"Total Return:        ${report.total_return:,.2f} ({report.total_return_pct:.2f}%)")
        print(f"Annualized Return:   {report.annualized_return:.2f}%")
        print(f"Avg Daily Return:    {report.avg_daily_return:.4f}%")

        print("\n--- Risk Metrics ---")
        print(f"Sharpe Ratio:        {report.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {report.sortino_ratio:.2f}")
        print(f"Calmar Ratio:        {report.calmar_ratio:.2f}")
        print(f"Max Drawdown:        ${report.max_drawdown:,.2f} ({report.max_drawdown_pct:.2f}%)")
        print(f"Max DD Duration:     {report.max_drawdown_duration} days")
        print(f"Volatility (ann.):   {report.volatility:.2f}%")

        print("\n--- Trade Statistics ---")
        print(f"Total Trades:        {report.total_trades}")
        print(f"Winning Trades:      {report.winning_trades}")
        print(f"Losing Trades:       {report.losing_trades}")
        print(f"Win Rate:            {report.win_rate:.2f}%")
        print(f"Profit Factor:       {report.profit_factor:.2f}")
        print(f"Avg Trade Return:    ${report.avg_trade_return:.2f} ({report.avg_trade_return_pct:.2f}%)")
        print(f"Best Trade:          ${report.best_trade:.2f}")
        print(f"Worst Trade:         ${report.worst_trade:.2f}")
        print(f"Avg Holding Period:  {report.avg_holding_period:.1f} days")

        print("=" * 60 + "\n")
