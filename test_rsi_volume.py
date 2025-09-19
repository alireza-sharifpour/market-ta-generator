#!/usr/bin/env python3
"""
Test script for RSI-enhanced volume analysis functionality.
This script demonstrates the new intelligent volume alerts with RSI context.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.volume_analyzer import VolumeAnalyzer
from app.utils.logging_config import setup_logging

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


async def test_rsi_enhanced_volume_analysis():
    """Test the RSI-enhanced volume analysis functionality."""
    
    print("🚀 Testing RSI-Enhanced Volume Analysis")
    print("=" * 60)
    
    # Test parameters
    pair = "btc_usdt"
    timeframe = "hour1"
    periods = 168  # 1 week of hourly data
    
    print(f"Testing RSI-enhanced volume analysis for {pair.upper()}")
    print(f"Timeframe: {timeframe}")
    print(f"Periods: {periods}")
    print(f"Method: Mean + 4×Std Dev + RSI Intelligence")
    print("-" * 60)
    
    try:
        # Initialize analyzer
        analyzer = VolumeAnalyzer()
        
        print("📊 Fetching data and performing RSI-enhanced analysis...")
        
        # Perform analysis
        result = await analyzer.analyze_pair(pair, timeframe, periods)
        
        print(f"✅ Analysis completed successfully!")
        print(f"⏰ Analysis timestamp: {result.analysis_timestamp}")
        print(f"📈 Data points: {len(result.data)}")
        print(f"🚨 Suspicious periods: {len(result.suspicious_periods)}")
        print(f"🎯 Confidence score: {result.confidence_score:.2%}")
        print("-" * 60)
        
        # Display RSI-enhanced metrics
        print("📊 RSI-ENHANCED ANALYSIS METRICS:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("-" * 60)
        
        # Display intelligent alerts
        if result.alerts:
            print("🚨 INTELLIGENT ALERTS:")
            for alert in result.alerts:
                level = alert.get('level', 'info').upper()
                message = alert.get('message', '')
                alert_type = alert.get('type', 'unknown')
                
                # Add emoji based on alert type
                emoji = ""
                if alert_type == "bearish_volume_spike":
                    emoji = "🐻"
                elif alert_type == "bullish_volume_spike":
                    emoji = "🐂"
                elif alert_type == "volume_spike":
                    emoji = "📊"
                elif alert_type == "summary":
                    emoji = "📋"
                
                print(f"  {emoji} [{level}] {message}")
                
                # Show additional details for RSI alerts
                if alert_type in ["bearish_volume_spike", "bullish_volume_spike"]:
                    if "avg_rsi" in alert:
                        print(f"      Average RSI: {alert['avg_rsi']:.1f}")
                    if "max_volume_ratio" in alert:
                        print(f"      Max Volume Ratio: {alert['max_volume_ratio']:.1f}x")
        else:
            print("✅ No alerts generated.")
        print("-" * 60)
        
        # Display suspicious periods with RSI context
        if result.suspicious_periods:
            print("🔍 SUSPICIOUS PERIODS WITH RSI CONTEXT:")
            for i, period in enumerate(result.suspicious_periods[:10]):  # Show first 10
                timestamp = period['timestamp']
                score = period.get('score', 0)
                volume = period.get('volume', 0)
                spike_ratio = period.get('volume_spike_ratio', 1.0)
                rsi = period.get('rsi', None)
                alert_type = period.get('alert_type', 'volume_spike')
                
                # Add emoji based on alert type
                emoji = ""
                if alert_type == "potential_market_top":
                    emoji = "🐻"
                elif alert_type == "potential_market_bottom":
                    emoji = "🐂"
                else:
                    emoji = "📊"
                
                rsi_info = f", RSI: {rsi:.1f}" if rsi is not None else ""
                print(f"  {i+1}. {emoji} {timestamp} - Score: {score}, Volume: {volume:,.0f}, Ratio: {spike_ratio:.1f}x{rsi_info}")
            
            if len(result.suspicious_periods) > 10:
                print(f"  ... and {len(result.suspicious_periods) - 10} more periods")
        else:
            print("✅ No suspicious periods detected.")
        print("-" * 60)
        
        # Summary of RSI-enhanced features
        bearish_count = len([sp for sp in result.suspicious_periods if "bearish_volume_spike" in sp["alerts"]])
        bullish_count = len([sp for sp in result.suspicious_periods if "bullish_volume_spike" in sp["alerts"]])
        standard_count = len([sp for sp in result.suspicious_periods if "mean_std_volume_spike" in sp["alerts"] and "bearish_volume_spike" not in sp["alerts"] and "bullish_volume_spike" not in sp["alerts"]])
        
        print("🎯 RSI-ENHANCED ALERT SUMMARY:")
        print(f"  🐻 Bearish alerts (potential tops): {bearish_count}")
        print(f"  🐂 Bullish alerts (potential bottoms): {bullish_count}")
        print(f"  📊 Standard volume spikes: {standard_count}")
        print(f"  📈 Total suspicious periods: {len(result.suspicious_periods)}")
        print("-" * 60)
        
        print("✅ RSI-enhanced volume analysis test completed successfully!")
        print("🎉 The system now provides intelligent volume alerts with RSI context!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during RSI-enhanced volume analysis: {str(e)}")
        logger.error(f"RSI-enhanced volume analysis test failed: {str(e)}", exc_info=True)
        return False


async def main():
    """Main test function."""
    success = await test_rsi_enhanced_volume_analysis()
    
    if success:
        print("🎉 RSI-enhanced volume analysis test completed successfully!")
        return 0
    else:
        print("❌ RSI-enhanced volume analysis test failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
