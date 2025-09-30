#!/usr/bin/env python3
"""
Standalone Volume Analysis Scheduler Service.
Runs the volume analysis scheduler as a standalone Python service.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from app.core.volume_scheduler import VolumeAnalysisScheduler
from app.core.batch_volume_analyzer import BatchVolumeAnalyzer
from app.utils.logging_config import setup_logging

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


class VolumeSchedulerService:
    """Standalone service for volume analysis scheduling."""
    
    def __init__(self):
        self.scheduler = VolumeAnalysisScheduler()
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def start_service(self, schedule_type: str = "interval", run_once: bool = False, test_alert: bool = False):
        """Start the volume analysis scheduler service."""
        try:
            if test_alert:
                logger.info("🚀 Testing Telegram Alert (Preview Mode)")
                await self._test_telegram_alert()
                return
            
            if run_once:
                logger.info("🚀 Running Volume Analysis Once (Test Mode)")
                await self._run_single_analysis()
                return
            
            logger.info("🚀 Starting Volume Analysis Scheduler Service")
            logger.info(f"Schedule type: {schedule_type}")
            
            # Start the scheduler
            self.scheduler.start_scheduler(schedule_type)
            
            # Log initial status
            status = self.scheduler.get_scheduler_status()
            logger.info(f"Scheduler status: {status}")
            
            # Show next run time in IRST
            if status.get("jobs"):
                for job in status["jobs"]:
                    if job.get("next_run_time"):
                        from datetime import datetime
                        import pytz
                        next_run = datetime.fromisoformat(job["next_run_time"].replace('Z', '+00:00'))
                        irst_tz = pytz.timezone('Asia/Tehran')
                        next_run_irst = next_run.astimezone(irst_tz)
                        logger.info(f"⏰ Next analysis run: {next_run_irst.strftime('%Y-%m-%d %H:%M:%S IRST')}")
            
            # Wait for shutdown signal
            logger.info("Service running... Press Ctrl+C to stop")
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Service error: {str(e)}", exc_info=True)
            raise
        finally:
            if not run_once:  # Only stop service if not running once
                await self.stop_service()
    
    async def _run_single_analysis(self):
        """Run volume analysis once and exit."""
        try:
            logger.info("🔄 Starting single volume analysis run...")
            
            # Initialize batch analyzer
            if not self.scheduler.batch_analyzer:
                self.scheduler.batch_analyzer = BatchVolumeAnalyzer()
            
            # Run analysis
            summary = await self.scheduler.batch_analyzer.analyze_all_pairs()
            
            # Log summary
            batch_summary = summary["batch_analysis_summary"]
            logger.info(f"📊 Analysis Summary:")
            logger.info(f"  - Total pairs: {batch_summary['total_pairs']}")
            logger.info(f"  - Successful: {batch_summary['successful_pairs']}")
            logger.info(f"  - Failed: {batch_summary['failed_pairs']}")
            logger.info(f"  - Suspicious: {batch_summary['pairs_with_suspicious_periods']}")
            logger.info(f"  - Duration: {batch_summary['duration_seconds']:.2f}s")
            logger.info(f"  - Success rate: {batch_summary['success_rate']:.1f}%")
            
            # Log top suspicious pairs
            top_pairs = summary["top_suspicious_pairs"]
            if top_pairs:
                logger.info(f"🚨 Top suspicious pairs:")
                for i, pair in enumerate(top_pairs[:5], 1):
                    logger.info(f"  {i}. {pair['pair'].upper()}: {pair['suspicious_periods']} periods (confidence: {pair['confidence_score']:.2%})")
            
            # Send Telegram notification
            await self.scheduler.telegram_notifier.send_batch_summary(summary)
            
            logger.info("✅ Single volume analysis completed successfully")
            logger.info("🏁 Exiting (run-once mode)")
            
        except Exception as e:
            logger.error(f"❌ Single volume analysis failed: {str(e)}", exc_info=True)
            raise
    
    async def _test_telegram_alert(self):
        """Generate and send a test alert to Telegram."""
        try:
            logger.info("📊 Generating test alert with mock data...")
            
            # Create mock suspicious volume data
            mock_result = self._create_mock_volume_analysis_result()
            
            # Send the test alert to Telegram
            await self.scheduler.telegram_notifier.send_analysis_notification(mock_result)
            
            logger.info("✅ Test alert sent to Telegram successfully")
            logger.info("🏁 Exiting (test-alert mode)")
            
        except Exception as e:
            logger.error(f"❌ Test alert failed: {str(e)}", exc_info=True)
            raise
    
    def _create_mock_volume_analysis_result(self):
        """Create mock VolumeAnalysisResult for testing."""
        from app.core.volume_analyzer import VolumeAnalysisResult
        import pandas as pd
        from datetime import datetime, timedelta
        import numpy as np
        
        # Create mock result
        result = VolumeAnalysisResult()
        result.pair = "btc_usdt"
        result.timeframe = "minute5"
        result.analysis_timestamp = datetime.now()
        result.confidence_score = 0.85
        
        # Create mock DataFrame with realistic data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=4),
            end=datetime.now(),
            freq='5min'
        )
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For consistent test data
        base_price = 45000
        prices = []
        volumes = []
        
        for i in range(len(timestamps)):
            # Simulate price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            base_price *= (1 + price_change)
            prices.append(base_price)
            
            # Simulate volume with occasional spikes
            base_volume = 1000000
            if i == len(timestamps) - 1:  # Last candle has suspicious volume
                volume_multiplier = 8.5  # High suspicious volume
            else:
                volume_multiplier = np.random.uniform(0.5, 2.0)
            volumes.append(base_volume * volume_multiplier)
        
        # Create DataFrame
        df_data = {
            'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'High': [p * np.random.uniform(1.0, 1.01) for p in prices],
            'Low': [p * np.random.uniform(0.99, 1.0) for p in prices],
            'Close': prices,
            'Volume': volumes
        }
        
        result.data = pd.DataFrame(df_data, index=timestamps)
        
        # Add technical indicators
        result.data['RSI_14'] = np.random.uniform(20, 80, len(timestamps))
        result.data['volume_mean'] = np.mean(volumes)
        result.data['volume_std'] = np.std(volumes)
        result.data['volume_threshold_low'] = result.data['volume_mean'] + 2.0 * result.data['volume_std']
        result.data['volume_threshold_medium'] = result.data['volume_mean'] + 4.0 * result.data['volume_std']
        result.data['volume_threshold_high'] = result.data['volume_mean'] + 6.0 * result.data['volume_std']
        
        # Create mock suspicious periods
        last_timestamp = timestamps[-1]
        last_volume = volumes[-1]
        last_price = prices[-1]
        last_rsi = result.data['RSI_14'].iloc[-1]
        
        # Calculate volume spike ratio
        volume_spike_ratio = last_volume / result.data['volume_mean'].iloc[-1]
        
        # Determine severity and alert type based on RSI
        if last_rsi > 70:
            severity = "high"
            alert_type = "potential_market_top_high"
            alerts = ["bearish_volume_spike_high"]
            emoji = "🐻"
        elif last_rsi < 30:
            severity = "medium"
            alert_type = "potential_market_bottom_medium"
            alerts = ["bullish_volume_spike_medium"]
            emoji = "🐂"
        else:
            severity = "low"
            alert_type = "volume_spike_low"
            alerts = ["volume_spike_low"]
            emoji = "📊"
        
        result.suspicious_periods = [{
            "timestamp": last_timestamp,
            "index": len(timestamps) - 1,
            "alerts": alerts,
            "alert_type": alert_type,
            "score": 4 if severity == "high" else 3 if severity == "medium" else 2,
            "severity": severity,
            "volume": last_volume,
            "volume_mean": result.data['volume_mean'].iloc[-1],
            "volume_spike_threshold": result.data[f'volume_threshold_{severity}'].iloc[-1],
            "volume_spike_ratio": volume_spike_ratio,
            "price": last_price,
            "rsi": last_rsi,
        }]
        
        # Create mock metrics
        result.metrics = {
            "total_periods": len(timestamps),
            "suspicious_periods": len(result.suspicious_periods),
            "suspicious_percentage": (len(result.suspicious_periods) / len(timestamps)) * 100,
            "volume_spike_ratio": volume_spike_ratio,
            "rsi_value": last_rsi,
            "price_change_percent": ((last_price - prices[0]) / prices[0]) * 100
        }
        
        # Create mock alerts (should be a list of alert dictionaries)
        result.alerts = [{
            "type": "summary",
            "level": "critical" if severity == "high" else "warning" if severity == "medium" else "info",
            "message": f"Volume spike detected with {severity} severity",
            "severity": severity,
            "alert_type": alert_type,
            "confidence": result.confidence_score,
            "rsi_enhanced": True,
            "volume_spike_ratio": volume_spike_ratio
        }]
        
        logger.info(f"📈 Created mock data for {result.pair.upper()}")
        logger.info(f"   - {len(result.suspicious_periods)} suspicious period(s)")
        logger.info(f"   - Severity: {severity.upper()}")
        logger.info(f"   - Alert type: {alert_type}")
        logger.info(f"   - RSI: {last_rsi:.1f}")
        logger.info(f"   - Volume spike: {result.metrics['volume_spike_ratio']:.1f}x")
        
        return result
    
    async def stop_service(self):
        """Stop the volume analysis scheduler service."""
        try:
            logger.info("🛑 Stopping Volume Analysis Scheduler Service")
            
            if self.scheduler.is_running:
                self.scheduler.stop_scheduler()
            
            logger.info("✅ Service stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


async def main():
    """Main function for the scheduler service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Volume Analysis Scheduler Service")
    parser.add_argument(
        "--schedule-type",
        choices=["interval", "cron"],
        default="interval",
        help="Schedule type: interval (every N minutes) or cron (cron-like)"
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=5,
        help="Interval in minutes for interval schedule type"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run volume analysis once and exit (for testing)"
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Generate and send a test alert to Telegram (for testing notifications)"
    )
    
    args = parser.parse_args()
    
    # Create service
    service = VolumeSchedulerService()
    
    # Update scheduler config if needed
    if args.schedule_type == "interval":
        service.scheduler.schedule_config["interval_minutes"] = args.interval_minutes
    
    try:
        # Start the service
        await service.start_service(args.schedule_type, run_once=args.run_once, test_alert=args.test_alert)
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
