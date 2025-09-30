"""
Telegram Notifier for sending volume analysis results to Telegram channels.
For suspicious volume analysis, sends only pictures with text in caption.
For non-suspicious results, sends text summaries if enabled.
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
        
        logger.info(f"TelegramNotifier initialization - enabled: {self.config['enabled']}, token present: {bool(self.config['bot_token'])}")
        
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
            logger.info(f"✅ TelegramNotifier initialized successfully with timeouts: connect={self.config.get('connect_timeout', 30)}s, read={self.config.get('read_timeout', 60)}s, write={self.config.get('write_timeout', 60)}s")
        else:
            logger.warning("❌ Telegram notifications disabled - missing token or disabled in config")
            logger.warning(f"   Config enabled: {self.config['enabled']}")
            logger.warning(f"   Token present: {bool(self.config['bot_token'])}")
            logger.warning(f"   Channel ID: {self.config.get('channel_id', 'Not set')}")
    
    async def send_analysis_notification(self, result: VolumeAnalysisResult) -> bool:
        """
        Send analysis notification to Telegram channel.
        For suspicious volume analysis, only sends picture with text in caption.
        
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
            
            # For suspicious volume analysis, send only picture with caption
            if result.suspicious_periods:
                await self._send_chart_with_caption(result)
            elif self.config["send_summary"]:
                # Only send text message if no suspicious periods and summary is enabled
                await self._send_summary_message(result)
            
            logger.info(f"✅ Telegram notification sent for {result.pair}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram notification for {result.pair}: {str(e)}")
            return False
    
    async def send_batch_summary(self, summary: Dict[str, Any], timeframe_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send batch analysis summary to Telegram channel.
        
        Args:
            summary: Batch analysis summary
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.bot or not self.config["enabled"]:
            logger.warning("Telegram notifications disabled - bot not initialized or disabled in config")
            logger.warning(f"Bot initialized: {self.bot is not None}, Config enabled: {self.config['enabled']}")
            return False
        
        try:
            logger.info("📤 Preparing to send batch summary to Telegram...")
            message = self._format_batch_summary(summary)
            logger.info(f"📝 Message prepared (length: {len(message)} chars)")
            
            await self.bot.send_message(
                chat_id=self.config["channel_id"],
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Batch summary sent to Telegram successfully")
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
    
    async def _send_chart_with_caption(self, result: VolumeAnalysisResult):
        """Send chart image with formatted analysis text as caption."""
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
            
            # Format the analysis message as caption
            caption = self._format_analysis_message(result)
            
            # Send photo with caption
            await self.bot.send_photo(
                chat_id=self.config["channel_id"],
                photo=chart_io,
                caption=caption,
                parse_mode='HTML'
            )
            
        except Exception as e:
            logger.error(f"Failed to send chart with caption: {str(e)}")
            # Fallback: send text message
            await self.bot.send_message(
                chat_id=self.config["channel_id"],
                text=f"📊 خطا در تولید نمودار {result.pair.upper()}\nخطا: {str(e)}",
                parse_mode='HTML'
            )
    
    async def _send_chart_image(self, result: VolumeAnalysisResult):
        """Send chart image (legacy method for backward compatibility)."""
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
                text=f"📊 خطا در تولید نمودار {result.pair.upper()}\nخطا: {str(e)}",
                parse_mode='HTML'
            )
    
    def _format_analysis_message(self, result: VolumeAnalysisResult) -> str:
        """Format analysis result into Telegram message in Farsi."""
        
        # Emojis based on confidence
        if result.confidence_score >= 0.8:
            confidence_emoji = "🚨"
        elif result.confidence_score >= 0.6:
            confidence_emoji = "⚠️"
        else:
            confidence_emoji = "ℹ️"
        
        # Count alert types
        bearish_count = len([sp for sp in result.suspicious_periods if any("bearish_volume_spike" in alert for alert in sp["alerts"])])
        bullish_count = len([sp for sp in result.suspicious_periods if any("bullish_volume_spike" in alert for alert in sp["alerts"])])
        
        # Get severity level
        severity_counts = {}
        for period in result.suspicious_periods:
            severity = period.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determine primary severity
        if severity_counts:
            severity_levels = {"low": 1, "medium": 2, "high": 3}
            primary_severity = max(severity_counts.keys(), key=lambda x: severity_levels.get(x, 0))
        else:
            primary_severity = "none"
        
        # Build short Farsi message
        if result.suspicious_periods:
            # Alert message
            severity_text = {"high": "بالا", "medium": "متوسط", "low": "پایین"}.get(primary_severity, "نامشخص")
            
            # Convert timeframe to Farsi numbers
            timeframe_farsi = result.timeframe.replace("1h", "۱ ساعت").replace("4h", "۴ ساعت").replace("1d", "۱ روز").replace("5m", "۵ دقیقه").replace("15m", "۱۵ دقیقه").replace("30m", "۳۰ دقیقه")
            
            # Main message with emoji
            message = f"ℹ️ حجم مشکوک\n\n💱 {result.pair.upper()} | ⏰ {result.timeframe}"
            
            # Add severity level
            message += f"\nسطح: {severity_text}"
            
            # Add specific alert type if available
            if bearish_count > 0 and bullish_count == 0:
                message += f"\n🔴🐻 احتمال دامپ"
            elif bullish_count > 0 and bearish_count == 0:
                message += f"\n🟢🐂 احتمال پامپ"
            elif bearish_count > 0 and bullish_count > 0:
                message += f"\n🔴🐻 احتمال دامپ\n🟢🐂 احتمال پامپ"
                
        else:
            # No alerts message
            message = f"""✅ <b>تحلیل حجم</b>

💱 <b>{result.pair.upper()}</b> | ⏰ {result.timeframe}
🎯 <b>اعتماد:</b> {result.confidence_score:.0%}

✅ <b>هیچ فعالیت مشکوکی یافت نشد</b>"""
        
        return message
    
    def _format_batch_summary(self, summary: Dict[str, Any]) -> str:
        """Format batch summary into Telegram message in Farsi."""
        
        batch_summary = summary["batch_analysis_summary"]
        
        message = f"""📊 <b>خلاصه تحلیل دسته‌ای</b>

⏰ <b>تکمیل:</b> {datetime.fromisoformat(batch_summary['end_time']).strftime('%H:%M:%S')}
⚡ <b>مدت:</b> {batch_summary['duration_seconds']:.0f}ثانیه

📈 <b>نتایج:</b>
• کل جفت‌ها: {batch_summary['total_pairs']}
• موفق: {batch_summary['successful_pairs']}
• نرخ موفقیت: {batch_summary['success_rate']:.0f}%

🚨 <b>فعالیت مشکوک:</b>
• جفت‌های هشداردار: {batch_summary['pairs_with_suspicious_periods']}
• نرخ مشکوک: {batch_summary['suspicious_rate']:.0f}%"""

        # Show top suspicious pairs (only top 3)
        top_pairs = summary.get("top_suspicious_pairs", [])
        if top_pairs:
            message += f"\n\n🔝 <b>جفت‌های مشکوک:</b>"
            for i, pair in enumerate(top_pairs[:3], 1):
                message += f"\n{i}. {pair['pair'].upper()}: {pair['suspicious_periods']} هشدار"
        
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
                text="🤖 <b>تست ربات تلگرام</b>\n\nربات متصل است و آماده ارسال هشدارهای تحلیل حجم!",
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
