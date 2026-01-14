"""Notification system for trade reports via Discord webhook."""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeReportData:
    """Data for trade report."""
    report_period: str
    total_value: float
    initial_capital: float
    total_return_pct: float
    cash_balance: float
    num_positions: int
    positions: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]
    category_breakdown: Dict[str, Dict[str, Any]]
    top_gainers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    risk_status: Dict[str, Any]


class DiscordNotifier:
    """
    Send trading notifications and reports to Discord via webhook.

    Uses Discord's embed format for rich formatting.
    """

    def __init__(self, webhook_url: str):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

        if not self.enabled:
            logger.warning("Discord webhook URL not configured - notifications disabled")

    def _send_webhook(self, payload: Dict) -> bool:
        """Send payload to Discord webhook."""
        if not self.enabled:
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            logger.info("Discord notification sent successfully")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_trade_report(self, report: TradeReportData) -> bool:
        """
        Send comprehensive 12-hour trade report to Discord.

        Args:
            report: TradeReportData with all report information

        Returns:
            True if sent successfully
        """
        # Determine overall status color
        if report.total_return_pct >= 5:
            color = 0x00FF00  # Green - great
        elif report.total_return_pct >= 0:
            color = 0x90EE90  # Light green - positive
        elif report.total_return_pct >= -5:
            color = 0xFFA500  # Orange - small loss
        else:
            color = 0xFF0000  # Red - significant loss

        # Build position fields
        position_text = ""
        if report.positions:
            for pos in report.positions[:10]:  # Limit to 10
                pnl_emoji = "🟢" if pos["unrealized_pnl"] >= 0 else "🔴"
                pnl_sign = "+" if pos["unrealized_pnl"] >= 0 else ""
                position_text += (
                    f"{pnl_emoji} **{pos['symbol']}** | "
                    f"${pos['current_value']:,.2f} | "
                    f"{pnl_sign}{pos['unrealized_pnl_pct']:.1f}%\n"
                )
        else:
            position_text = "No open positions"

        # Build recent trades text
        trades_text = ""
        if report.recent_trades:
            for trade in report.recent_trades[:5]:  # Last 5 trades
                trade_emoji = "🟢" if trade["pnl"] >= 0 else "🔴"
                pnl_sign = "+" if trade["pnl"] >= 0 else ""
                trades_text += (
                    f"{trade_emoji} **{trade['symbol']}** {trade['side'].upper()} | "
                    f"{pnl_sign}${trade['pnl']:.2f} ({pnl_sign}{trade['pnl_pct']:.1f}%)\n"
                )
        else:
            trades_text = "No trades in this period"

        # Build category breakdown
        category_text = ""
        for cat, data in report.category_breakdown.items():
            if data["value"] > 0:
                cat_emoji = "📊"
                category_text += (
                    f"{cat_emoji} **{cat}**: ${data['value']:,.2f} "
                    f"({data['allocation']:.1f}%) - {data['positions']} positions\n"
                )

        # Top gainers/losers
        gainers_text = ""
        if report.top_gainers:
            for g in report.top_gainers[:3]:
                gainers_text += f"🚀 **{g['symbol']}**: +{g['pnl_pct']:.1f}%\n"
        else:
            gainers_text = "None"

        losers_text = ""
        if report.top_losers:
            for l in report.top_losers[:3]:
                losers_text += f"📉 **{l['symbol']}**: {l['pnl_pct']:.1f}%\n"
        else:
            losers_text = "None"

        # Risk status
        risk_emoji = "✅" if not report.risk_status.get("trading_halted") else "🚨"
        drawdown = report.risk_status.get("current_drawdown", 0) * 100

        # Build the embed
        embed = {
            "title": f"📈 Trading Report - {report.report_period}",
            "description": f"**Portfolio Performance Summary**",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "💰 Portfolio Value",
                    "value": f"**${report.total_value:,.2f}**",
                    "inline": True
                },
                {
                    "name": "📊 Total Return",
                    "value": f"**{'+' if report.total_return_pct >= 0 else ''}{report.total_return_pct:.2f}%**",
                    "inline": True
                },
                {
                    "name": "💵 Cash Available",
                    "value": f"${report.cash_balance:,.2f}",
                    "inline": True
                },
                {
                    "name": f"📍 Open Positions ({report.num_positions})",
                    "value": position_text[:1024],  # Discord limit
                    "inline": False
                },
                {
                    "name": "🔄 Recent Trades",
                    "value": trades_text[:1024] if trades_text else "No trades",
                    "inline": False
                },
                {
                    "name": "📁 Category Breakdown",
                    "value": category_text[:1024] if category_text else "No allocations",
                    "inline": False
                },
                {
                    "name": "🏆 Top Gainers",
                    "value": gainers_text,
                    "inline": True
                },
                {
                    "name": "📉 Top Losers",
                    "value": losers_text,
                    "inline": True
                },
                {
                    "name": f"{risk_emoji} Risk Status",
                    "value": f"Drawdown: {drawdown:.1f}% | Consecutive Losses: {report.risk_status.get('consecutive_losses', 0)}",
                    "inline": False
                }
            ],
            "footer": {
                "text": "Crypto ML Trading Bot | Paper Trading Mode"
            }
        }

        payload = {
            "username": "Crypto ML Trader",
            "avatar_url": "https://i.imgur.com/4M34hi2.png",
            "embeds": [embed]
        }

        return self._send_webhook(payload)

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        category: str,
        quantity: float,
        price: float,
        confidence: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None
    ) -> bool:
        """
        Send instant trade alert to Discord.

        Args:
            action: BUY or SELL
            symbol: Coin symbol
            category: Coin category
            quantity: Trade quantity
            price: Execution price
            confidence: Model confidence
            pnl: Realized P&L (for sells)
            pnl_pct: Realized P&L percentage (for sells)
        """
        if action.upper() == "BUY":
            color = 0x00FF00  # Green
            emoji = "🟢"
            title = f"{emoji} BUY Signal Executed"
        else:
            color = 0xFF6B6B if pnl and pnl < 0 else 0x00FF00
            emoji = "🔴" if pnl and pnl < 0 else "🟢"
            title = f"{emoji} SELL Signal Executed"

        fields = [
            {"name": "Coin", "value": f"**{symbol}**", "inline": True},
            {"name": "Category", "value": category, "inline": True},
            {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True},
            {"name": "Quantity", "value": f"{quantity:.6f}", "inline": True},
            {"name": "Price", "value": f"${price:,.2f}", "inline": True},
            {"name": "Value", "value": f"${quantity * price:,.2f}", "inline": True},
        ]

        if pnl is not None:
            pnl_sign = "+" if pnl >= 0 else ""
            fields.append({
                "name": "Realized P&L",
                "value": f"{pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)",
                "inline": False
            })

        embed = {
            "title": title,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": fields,
            "footer": {"text": "Crypto ML Trading Bot"}
        }

        payload = {
            "username": "Crypto ML Trader",
            "embeds": [embed]
        }

        return self._send_webhook(payload)

    def send_risk_alert(self, alert_type: str, message: str, details: Dict[str, Any]) -> bool:
        """
        Send risk management alert to Discord.

        Args:
            alert_type: Type of alert (DRAWDOWN, DAILY_LOSS, CONSECUTIVE_LOSSES, etc.)
            message: Alert message
            details: Additional details
        """
        embed = {
            "title": f"🚨 RISK ALERT: {alert_type}",
            "description": message,
            "color": 0xFF0000,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": k, "value": str(v), "inline": True}
                for k, v in details.items()
            ],
            "footer": {"text": "Crypto ML Trading Bot - Risk Management"}
        }

        payload = {
            "username": "Crypto ML Trader",
            "embeds": [embed]
        }

        return self._send_webhook(payload)

    def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send quick daily summary."""
        pnl = summary.get("daily_pnl", 0)
        pnl_pct = summary.get("daily_pnl_pct", 0)

        color = 0x00FF00 if pnl >= 0 else 0xFF0000
        emoji = "📈" if pnl >= 0 else "📉"
        pnl_sign = "+" if pnl >= 0 else ""

        embed = {
            "title": f"{emoji} Daily Summary",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Daily P&L", "value": f"{pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%)", "inline": True},
                {"name": "Portfolio Value", "value": f"${summary.get('total_value', 0):,.2f}", "inline": True},
                {"name": "Open Positions", "value": str(summary.get("num_positions", 0)), "inline": True},
                {"name": "Trades Today", "value": str(summary.get("trades_today", 0)), "inline": True},
            ]
        }

        payload = {
            "username": "Crypto ML Trader",
            "embeds": [embed]
        }

        return self._send_webhook(payload)

    def send_startup_notification(
        self,
        config: Dict[str, Any],
        num_models: int,
        num_coins: int
    ) -> bool:
        """
        Send notification when the trading bot starts.

        Args:
            config: Trading configuration details
            num_models: Number of loaded ML models
            num_coins: Number of tradeable coins
        """
        embed = {
            "title": "🚀 Crypto ML Trading Bot Started",
            "description": "The trading system is now online and monitoring the markets.",
            "color": 0x00BFFF,  # Deep sky blue
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Mode", "value": config.get("mode", "Paper Trading"), "inline": True},
                {"name": "Initial Capital", "value": f"${config.get('initial_capital', 10000):,.2f}", "inline": True},
                {"name": "Models Loaded", "value": str(num_models), "inline": True},
                {"name": "Coins Tracked", "value": str(num_coins), "inline": True},
                {"name": "Max Position", "value": f"{config.get('max_position_pct', 0.2):.0%}", "inline": True},
                {"name": "Max Exposure", "value": f"{config.get('max_total_exposure', 0.8):.0%}", "inline": True},
                {"name": "Max Drawdown", "value": f"{config.get('max_drawdown_pct', 0.15):.0%}", "inline": True},
                {"name": "Report Interval", "value": "Every 12 hours", "inline": True},
            ],
            "footer": {"text": "Crypto ML Trading Bot | Paper Trading Mode"}
        }

        payload = {
            "username": "Crypto ML Trader",
            "avatar_url": "https://i.imgur.com/4M34hi2.png",
            "embeds": [embed]
        }

        return self._send_webhook(payload)

    def send_shutdown_notification(self, reason: str = "Manual shutdown") -> bool:
        """Send notification when the trading bot stops."""
        embed = {
            "title": "🔴 Crypto ML Trading Bot Stopped",
            "description": f"The trading system has been shut down.",
            "color": 0xFF6B6B,  # Light red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
            ],
            "footer": {"text": "Crypto ML Trading Bot"}
        }

        payload = {
            "username": "Crypto ML Trader",
            "embeds": [embed]
        }

        return self._send_webhook(payload)


class ReportGenerator:
    """Generate trading reports from portfolio and trade data."""

    def __init__(
        self,
        engine,  # PaperTradingEngine
        portfolio_manager,  # MultiCoinPortfolioManager
        risk_manager,  # RiskManager
        registry  # CoinRegistry
    ):
        self.engine = engine
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.registry = registry

    def generate_report(
        self,
        prices: Dict[str, float],
        hours: int = 12
    ) -> TradeReportData:
        """
        Generate comprehensive trade report.

        Args:
            prices: Current prices for all coins
            hours: Report period in hours

        Returns:
            TradeReportData ready for notification
        """
        # Get portfolio state
        state = self.portfolio_manager.get_portfolio_state(prices)

        # Calculate period
        now = datetime.now()
        period_start = now - timedelta(hours=hours)
        report_period = f"{period_start.strftime('%m/%d %H:%M')} - {now.strftime('%m/%d %H:%M')}"

        # Format positions
        positions = []
        for coin_id, pos_data in state.positions.items():
            coin = self.registry.get_coin(coin_id)
            positions.append({
                "symbol": coin.symbol if coin else coin_id.upper(),
                "category": pos_data["category"],
                "quantity": pos_data["quantity"],
                "entry_price": pos_data["entry_price"],
                "current_price": pos_data["current_price"],
                "current_value": pos_data["quantity"] * pos_data["current_price"],
                "unrealized_pnl": pos_data["unrealized_pnl"],
                "unrealized_pnl_pct": pos_data["unrealized_pnl_pct"]
            })

        # Sort by P&L for top gainers/losers
        sorted_positions = sorted(positions, key=lambda x: x["unrealized_pnl_pct"], reverse=True)
        top_gainers = [p for p in sorted_positions if p["unrealized_pnl_pct"] > 0][:3]
        top_losers = [p for p in sorted_positions if p["unrealized_pnl_pct"] < 0][-3:][::-1]

        # Get recent trades
        recent_trades = []
        for trade in self.engine.trades[-10:]:  # Last 10 trades
            trade_time = trade.get("timestamp")
            if trade_time and trade_time >= period_start:
                coin = self.registry.get_coin(trade["symbol"])
                recent_trades.append({
                    "symbol": coin.symbol if coin else trade["symbol"].upper(),
                    "side": trade["side"],
                    "quantity": trade["quantity"],
                    "pnl": trade["pnl"],
                    "pnl_pct": trade.get("pnl_pct", 0),
                    "timestamp": trade_time
                })

        # Category breakdown
        category_breakdown = {}
        for category, alloc in state.allocations.items():
            if alloc.current_value > 0 or self.registry.get_category_config(category).enabled:
                category_breakdown[category.value] = {
                    "value": alloc.current_value,
                    "allocation": alloc.current_allocation * 100,
                    "positions": len(alloc.positions)
                }

        # Risk status
        risk_status = self.risk_manager.get_status(
            state.total_value,
            state.num_positions
        )

        # Calculate total return
        initial = self.engine.initial_capital
        total_return_pct = (state.total_value / initial - 1) * 100

        return TradeReportData(
            report_period=report_period,
            total_value=state.total_value,
            initial_capital=initial,
            total_return_pct=total_return_pct,
            cash_balance=state.cash_balance,
            num_positions=state.num_positions,
            positions=positions,
            recent_trades=recent_trades,
            category_breakdown=category_breakdown,
            top_gainers=top_gainers,
            top_losers=top_losers,
            risk_status=risk_status
        )
