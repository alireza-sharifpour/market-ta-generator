"""
Telegram Notifier for sending volume analysis results to Telegram channels.
Sends analysis summaries, alerts, and chart images.
"""

import asyncio
import logging
import io
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import telegram
from telegram import Bot
from telegram.error import TelegramError

from app.core.volume_analyzer import VolumeAnalysisResult
from app.core.volume_chart_generator import VolumeChartGenerator
from app.config import TELEGRAM_CONFIG

# Set up logging
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send volume analysis notifications to Telegram channels."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Telegram notifier."""
        self.config = config or TELEGRAM_CONFIG
        self.bot = None
        self.chart_generator = VolumeChartGenerator()
        
        if self.config["enabled"] and self.config["bot_token"]:
            # Create bot with configurable timeout settings
            self.bot = Bot(
                token=self.config["bot_token"],
                request=telegram.request.HTTPXRequest(
                    connection_pool_size=self.config.get("connection_pool_size", 8),
                    read_timeout=self.config.get("read_timeout", 60),
                    write_timeout=self.config.get("write_timeout", 60),
                    connect_timeout=self.config.get("connect_timeout", 30),
                )
            )
            logger.info(f"TelegramNotifier initialized with timeouts: connect={self.config.get('connect_timeout', 30)}s, read={self.config.get('read_timeout', 60)}s, write={self.config.get('write_timeout', 60)}s")
        else:
            logger.warning("Telegram notifications disabled - missing token or disabled in config")
    
    async def send_analysis_notification(self, result: VolumeAnalysisResult) -> bool:
        """
        Send analysis notification to Telegram channel.
        
        Args:
            result: VolumeAnalysisResult containing analysis data
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot or not self.config["enabled"]:
            logger.debug("Telegram notifications disabled")
            return False
        
        try:
            # Check if we should send (only alerts or always)
            if self.config["send_alerts_only"] and not result.suspicious_periods:
                logger.debug(f"No alerts for {result.pair}, skipping notification")
                return True
            
            # Send summary message
            if self.config["send_summary"]:
                await self._send_summary_message(result)
            
            # Send chart image if suspicious periods found
            if self.config["send_charts"] and result.suspicious_periods:
                await self._send_chart_image(result)
            
            logger.info(f"✅ Telegram notification sent for {result.pair}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram notification for {result.pair}: {str(e)}")
            return False
    
    async def send_batch_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Send batch analysis summary to Telegram channel.
        
        Args:
            summary: Batch analysis summary
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot or not self.config["enabled"]:
            logger.debug("Telegram notifications disabled")
            return False
        
        try:
            message = self._format_batch_summary(summary)
            
            await self.bot.send_message(
                chat_id=self.config["channel_id"],
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Batch summary sent to Telegram")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send batch summary to Telegram: {str(e)}")
            return False
    
    async def _send_summary_message(self, result: VolumeAnalysisResult):
        """Send formatted summary message."""
        message = self._format_analysis_message(result)
        
        await self.bot.send_message(
            chat_id=self.config["channel_id"],
            text=message,
            parse_mode='HTML'
        )
    
    async def _send_chart_image(self, result: VolumeAnalysisResult):
        """Send chart image."""
        try:
            # Generate chart as base64 PNG
            chart_base64 = self.chart_generator.create_analysis_chart_base64(result)
            
            # Remove data URL prefix if present
            if chart_base64.startswith('data:image/png;base64,'):
                chart_base64 = chart_base64.replace('data:image/png;base64,', '')
            
            # Convert base64 to bytes
            chart_bytes = base64.b64decode(chart_base64)
            
            # Create file-like object
            chart_io = io.BytesIO(chart_bytes)
            chart_io.name = f"{result.pair}_{result.timeframe}_analysis.png"
            
            # Send photo
            await self.bot.send_photo(
                chat_id=self.config["channel_id"],
                photo=chart_io,
                caption=f"📊 RSI-Enhanced Volume Analysis Chart\n{result.pair.upper()} - {result.timeframe}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send chart image: {str(e)}")
            # Fallback: send text message
            await self.bot.send_message(
                chat_id=self.config["channel_id"],
                text=f"📊 Chart generation failed for {result.pair.upper()}\nError: {str(e)}",
                parse_mode='HTML'
            )
    
    def _format_analysis_message(self, result: VolumeAnalysisResult) -> str:
        """Format analysis result into Telegram message."""
        
        # Emojis based on confidence
        if result.confidence_score >= 0.8:
            confidence_emoji = "🚨"
        elif result.confidence_score >= 0.6:
            confidence_emoji = "⚠️"
        else:
            confidence_emoji = "ℹ️"
        
        # Count alert types
        bearish_count = len([sp for sp in result.suspicious_periods if "bearish_volume_spike" in sp["alerts"]])
        bullish_count = len([sp for sp in result.suspicious_periods if "bullish_volume_spike" in sp["alerts"]])
        standard_count = len([sp for sp in result.suspicious_periods if "mean_std_volume_spike" in sp["alerts"] and "bearish_volume_spike" not in sp["alerts"] and "bullish_volume_spike" not in sp["alerts"]])
        
        # Build message
        message = f"""{confidence_emoji} <b>Volume Analysis Alert</b>
        
💱 <b>Pair:</b> {result.pair.upper()}
⏰ <b>Timeframe:</b> {result.timeframe}
📅 <b>Analysis Time:</b> {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S IRST')}
🎯 <b>Confidence:</b> {result.confidence_score:.1%}

📊 <b>Results:</b>
• Total Periods: {result.metrics.get('total_periods', 0)}
• Suspicious Periods: {len(result.suspicious_periods)}
• Suspicious Rate: {result.metrics.get('suspicious_percentage', 0):.1f}%"""

        if result.suspicious_periods:
            message += f"\n\n🚨 <b>Alert Breakdown:</b>"
            
            if bearish_count > 0:
                message += f"\n🐻 Bearish Alerts: {bearish_count} (Potential Market Tops)"
            
            if bullish_count > 0:
                message += f"\n🐂 Bullish Alerts: {bullish_count} (Potential Market Bottoms)"
            
            if standard_count > 0:
                message += f"\n📊 Standard Volume Spikes: {standard_count}"
            
            # Show max spike ratio
            max_ratio = result.metrics.get('max_spike_ratio', 0)
            if max_ratio > 0:
                message += f"\n📈 Max Volume Spike: {max_ratio:.1f}x threshold"
        else:
            message += f"\n\n✅ <b>No suspicious volume activity detected</b>"
        
        # Add alerts summary
        if result.alerts:
            critical_alerts = [a for a in result.alerts if a.get('level') == 'critical']
            if critical_alerts:
                message += f"\n\n🚨 <b>Critical Alerts:</b> {len(critical_alerts)}"
        
        return message
    
    def _format_batch_summary(self, summary: Dict[str, Any]) -> str:
        """Format batch summary into Telegram message."""
        
        batch_summary = summary["batch_analysis_summary"]
        
        message = f"""📊 <b>Batch Volume Analysis Summary</b>

⏰ <b>Completed:</b> {datetime.fromisoformat(batch_summary['end_time']).strftime('%Y-%m-%d %H:%M:%S IRST')}
⚡ <b>Duration:</b> {batch_summary['duration_seconds']:.1f}s

📈 <b>Analysis Results:</b>
• Total Pairs: {batch_summary['total_pairs']}
• Successful: {batch_summary['successful_pairs']}
• Failed: {batch_summary['failed_pairs']}
• Success Rate: {batch_summary['success_rate']:.1f}%

🚨 <b>Suspicious Activity:</b>
• Pairs with Alerts: {batch_summary['pairs_with_suspicious_periods']}
• Suspicious Rate: {batch_summary['suspicious_rate']:.1f}%"""

        if batch_summary['average_confidence_score'] > 0:
            message += f"\n• Avg Confidence: {batch_summary['average_confidence_score']:.1%}"
        
        # Show top suspicious pairs
        top_pairs = summary.get("top_suspicious_pairs", [])
        if top_pairs:
            message += f"\n\n🔝 <b>Top Suspicious Pairs:</b>"
            for i, pair in enumerate(top_pairs[:5], 1):
                message += f"\n{i}. {pair['pair'].upper()}: {pair['suspicious_periods']} periods ({pair['confidence_score']:.1%})"
        
        # Show failed pairs
        failed_pairs = summary.get("failed_pairs", [])
        if failed_pairs:
            message += f"\n\n❌ <b>Failed Pairs:</b> {len(failed_pairs)}"
            for pair in failed_pairs[:3]:
                message += f"\n• {pair['pair'].upper()}"
        
        return message
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection and permissions."""
        if not self.bot or not self.config["enabled"]:
            logger.error("Telegram bot not configured")
            return False
        
        try:
            # Test bot info
            me = await self.bot.get_me()
            logger.info(f"Bot connected: @{me.username}")
            
            # Test sending message
            await self.bot.send_message(
                chat_id=self.config["channel_id"],
                text="🤖 <b>Telegram Bot Test</b>\n\nBot is connected and ready to send volume analysis notifications!",
                parse_mode='HTML'
            )
            
            logger.info("✅ Telegram test message sent successfully")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram connection test failed: {str(e)}")
            return False
    
    async def get_channel_info(self) -> Optional[Dict]:
        """Get channel information for debugging."""
        if not self.bot or not self.config["enabled"]:
            return None
        
        try:
            chat = await self.bot.get_chat(self.config["channel_id"])
            return {
                "id": chat.id,
                "title": chat.title,
                "type": chat.type,
                "username": chat.username
            }
        except Exception as e:
            logger.error(f"Failed to get channel info: {str(e)}")
            return None


async def test_telegram_notifications():
    """Test function for Telegram notifications."""
    notifier = TelegramNotifier()
    
    if await notifier.test_connection():
        logger.info("Telegram notifications are working!")
        return True
    else:
        logger.error("Telegram notifications are not working!")
        return False


if __name__ == "__main__":
    asyncio.run(test_telegram_notifications())
