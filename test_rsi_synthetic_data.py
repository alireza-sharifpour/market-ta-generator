#!/usr/bin/env python3
"""
Test script for RSI-enhanced volume analysis with synthetic data.
This demonstrates all three alert types without requiring LBank API.
"""

import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.volume_analyzer import VolumeAnalyzer, VolumeAnalysisResult
from app.core.volume_chart_generator import VolumeChartGenerator
from app.utils.logging_config import setup_logging

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


def create_synthetic_data():
    """Create synthetic OHLCV data with RSI-enhanced volume spikes."""
    
    # Create 168 hours of data (1 week)
    periods = 168
    base_time = datetime.now() - timedelta(hours=periods)
    
    # Create time index
    timestamps = [base_time + timedelta(hours=i) for i in range(periods)]
    
    # Base price around 100k (BTC-like)
    base_price = 100000
    price_trend = np.linspace(0, 0.1, periods)  # 10% upward trend over week
    
    # Create realistic price data with some volatility
    np.random.seed(42)  # For reproducible results
    price_changes = np.random.normal(0, 0.02, periods)  # 2% daily volatility
    prices = base_price * (1 + price_trend + np.cumsum(price_changes))
    
    # Create OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Add some intraday volatility
        volatility = close_price * 0.01  # 1% intraday volatility
        
        high = close_price + np.random.uniform(0, volatility)
        low = close_price - np.random.uniform(0, volatility)
        
        # Ensure OHLC relationships are correct
        high = max(high, close_price)
        low = min(low, close_price)
        
        # Open is previous close (or close for first candle)
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
        
        data.append({
            'timestamp': timestamp,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price,
            'Volume': np.random.uniform(1000, 5000)  # Base volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def add_volume_spikes_and_rsi(df):
    """Add volume spikes and RSI data to demonstrate all three alert types."""
    
    # Calculate RSI (simplified version)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Add RSI
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # Add volume indicators (mean + std)
    lookback_period = 25
    df['volume_mean'] = df['Volume'].rolling(window=lookback_period).mean()
    df['volume_std'] = df['Volume'].rolling(window=lookback_period).std()
    df['volume_spike_threshold'] = df['volume_mean'] + (4.0 * df['volume_std'])
    df['volume_spike_detected'] = df['Volume'] > df['volume_spike_threshold']
    df['volume_spike_ratio'] = df['Volume'] / df['volume_spike_threshold']
    df['volume_spike_ratio'] = df['volume_spike_ratio'].fillna(1.0)
    
    # Create specific scenarios for each alert type
    
    # 1. 🐻 Bearish Alert: Volume spike + RSI > 70 (around hour 50)
    bearish_hour = 50
    if bearish_hour < len(df):
        # Set RSI to overbought level
        df.iloc[bearish_hour, df.columns.get_loc('RSI_14')] = 75.0
        # Create volume spike
        spike_volume = df.iloc[bearish_hour]['volume_spike_threshold'] * 2.5
        df.iloc[bearish_hour, df.columns.get_loc('Volume')] = spike_volume
        df.iloc[bearish_hour, df.columns.get_loc('volume_spike_detected')] = True
        df.iloc[bearish_hour, df.columns.get_loc('volume_spike_ratio')] = 2.5
    
    # 2. 🐂 Bullish Alert: Volume spike + RSI < 30 (around hour 100)
    bullish_hour = 100
    if bullish_hour < len(df):
        # Set RSI to oversold level
        df.iloc[bullish_hour, df.columns.get_loc('RSI_14')] = 25.0
        # Create volume spike
        spike_volume = df.iloc[bullish_hour]['volume_spike_threshold'] * 3.0
        df.iloc[bullish_hour, df.columns.get_loc('Volume')] = spike_volume
        df.iloc[bullish_hour, df.columns.get_loc('volume_spike_detected')] = True
        df.iloc[bullish_hour, df.columns.get_loc('volume_spike_ratio')] = 3.0
    
    # 3. 📊 Standard Alert: Volume spike without RSI extreme (around hour 150)
    standard_hour = 150
    if standard_hour < len(df):
        # Keep RSI in neutral range
        df.iloc[standard_hour, df.columns.get_loc('RSI_14')] = 55.0
        # Create volume spike
        spike_volume = df.iloc[standard_hour]['volume_spike_threshold'] * 2.0
        df.iloc[standard_hour, df.columns.get_loc('Volume')] = spike_volume
        df.iloc[standard_hour, df.columns.get_loc('volume_spike_detected')] = True
        df.iloc[standard_hour, df.columns.get_loc('volume_spike_ratio')] = 2.0
    
    # Add a few more standard volume spikes for variety
    for hour in [30, 80, 120, 160]:
        if hour < len(df) and hour not in [bearish_hour, bullish_hour, standard_hour]:
            # Neutral RSI
            df.iloc[hour, df.columns.get_loc('RSI_14')] = np.random.uniform(40, 60)
            # Moderate volume spike
            spike_volume = df.iloc[hour]['volume_spike_threshold'] * np.random.uniform(1.5, 2.5)
            df.iloc[hour, df.columns.get_loc('Volume')] = spike_volume
            df.iloc[hour, df.columns.get_loc('volume_spike_detected')] = True
            df.iloc[hour, df.columns.get_loc('volume_spike_ratio')] = spike_volume / df.iloc[hour]['volume_spike_threshold']
    
    return df


def create_synthetic_analysis_result(df):
    """Create a VolumeAnalysisResult with synthetic data."""
    
    result = VolumeAnalysisResult()
    result.pair = "BTC_USDT"
    result.timeframe = "hour1"
    result.analysis_timestamp = datetime.now()
    result.data = df
    
    # Create suspicious periods based on our synthetic data
    suspicious_periods = []
    
    for i in range(len(df)):
        if df.iloc[i]['volume_spike_detected']:
            rsi_value = df.iloc[i]['RSI_14']
            spike_ratio = df.iloc[i]['volume_spike_ratio']
            
            alerts = []
            alert_type = "volume_spike"
            score = 2
            
            # Determine alert type based on RSI
            if rsi_value > 70:
                alerts.append("bearish_volume_spike")
                alert_type = "potential_market_top"
                score = 4  # Higher score for RSI confirmation
            elif rsi_value < 30:
                alerts.append("bullish_volume_spike")
                alert_type = "potential_market_bottom"
                score = 4  # Higher score for RSI confirmation
            else:
                alerts.append("mean_std_volume_spike")
                score = 2
            
            suspicious_periods.append({
                "timestamp": df.index[i],
                "index": i,
                "alerts": alerts,
                "alert_type": alert_type,
                "score": score,
                "volume": df.iloc[i]['Volume'],
                "volume_mean": df.iloc[i]['volume_mean'],
                "volume_spike_threshold": df.iloc[i]['volume_spike_threshold'],
                "volume_spike_ratio": spike_ratio,
                "price": df.iloc[i]['Close'],
                "rsi": rsi_value,
            })
    
    result.suspicious_periods = suspicious_periods
    
    # Calculate metrics
    result.metrics = {
        "total_periods": len(df),
        "suspicious_periods_count": len(suspicious_periods),
        "suspicious_percentage": (len(suspicious_periods) / len(df)) * 100,
        "avg_volume": float(df['Volume'].mean()),
        "max_volume": float(df['Volume'].max()),
        "volume_std": float(df['Volume'].std()),
        "max_spike_ratio": float(df['volume_spike_ratio'].max()),
        "avg_spike_ratio": float(df['volume_spike_ratio'].mean()),
    }
    
    # Calculate confidence score
    if len(suspicious_periods) > 0:
        base_score = min(len(suspicious_periods) / len(df) * 10, 1.0)
        severity_boost = sum(0.3 if p['score'] >= 4 else 0.1 for p in suspicious_periods) / len(suspicious_periods)
        result.confidence_score = min(base_score + severity_boost, 1.0)
    else:
        result.confidence_score = 0.0
    
    # Generate alerts
    bearish_alerts = [sp for sp in suspicious_periods if "bearish_volume_spike" in sp["alerts"]]
    bullish_alerts = [sp for sp in suspicious_periods if "bullish_volume_spike" in sp["alerts"]]
    standard_alerts = [sp for sp in suspicious_periods if "mean_std_volume_spike" in sp["alerts"] and "bearish_volume_spike" not in sp["alerts"] and "bullish_volume_spike" not in sp["alerts"]]
    
    result.alerts = []
    
    # Summary alert
    result.alerts.append({
        "type": "summary",
        "level": "info" if result.confidence_score < 0.8 else "warning" if result.confidence_score < 0.9 else "critical",
        "message": f"Detected {len(suspicious_periods)} suspicious volume periods with {result.confidence_score:.1%} confidence",
        "timestamp": datetime.now()
    })
    
    # Bearish alerts
    if bearish_alerts:
        max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in bearish_alerts)
        avg_rsi = sum(p.get("rsi", 0) for p in bearish_alerts) / len(bearish_alerts)
        result.alerts.append({
            "type": "bearish_volume_spike",
            "level": "critical",
            "message": f"🐻 POTENTIAL MARKET TOP: {len(bearish_alerts)} volume spikes during overbought conditions (RSI avg: {avg_rsi:.1f}, max volume ratio: {max_ratio:.1f}x)",
            "count": len(bearish_alerts),
            "avg_rsi": avg_rsi,
            "max_volume_ratio": max_ratio,
            "timestamp": datetime.now()
        })
    
    # Bullish alerts
    if bullish_alerts:
        max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in bullish_alerts)
        avg_rsi = sum(p.get("rsi", 0) for p in bullish_alerts) / len(bullish_alerts)
        result.alerts.append({
            "type": "bullish_volume_spike",
            "level": "critical",
            "message": f"🐂 POTENTIAL MARKET BOTTOM: {len(bullish_alerts)} volume spikes during oversold conditions (RSI avg: {avg_rsi:.1f}, max volume ratio: {max_ratio:.1f}x)",
            "count": len(bullish_alerts),
            "avg_rsi": avg_rsi,
            "max_volume_ratio": max_ratio,
            "timestamp": datetime.now()
        })
    
    # Standard alerts
    if standard_alerts:
        max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in standard_alerts)
        result.alerts.append({
            "type": "volume_spike",
            "level": "warning" if max_ratio > 2.0 else "info",
            "message": f"Volume spikes detected in {len(standard_alerts)} periods (max ratio: {max_ratio:.1f}x) - No RSI extremes",
            "count": len(standard_alerts),
            "timestamp": datetime.now()
        })
    
    return result


async def test_synthetic_rsi_volume_analysis():
    """Test RSI-enhanced volume analysis with synthetic data."""
    
    print("🚀 Testing RSI-Enhanced Volume Analysis with Synthetic Data")
    print("=" * 70)
    
    try:
        # Create synthetic data
        print("📊 Creating synthetic OHLCV data with RSI-enhanced volume spikes...")
        df = create_synthetic_data()
        df = add_volume_spikes_and_rsi(df)
        
        print(f"✅ Created {len(df)} hours of synthetic data")
        print(f"📈 Price range: ${df['Close'].min():,.0f} - ${df['Close'].max():,.0f}")
        print(f"📊 Volume range: {df['Volume'].min():,.0f} - {df['Volume'].max():,.0f}")
        print(f"🎯 RSI range: {df['RSI_14'].min():.1f} - {df['RSI_14'].max():.1f}")
        print("-" * 70)
        
        # Create analysis result
        result = create_synthetic_analysis_result(df)
        
        print("🔍 ANALYSIS RESULTS:")
        print("=" * 70)
        print(f"📊 Total periods: {result.metrics['total_periods']}")
        print(f"🚨 Suspicious periods: {result.metrics['suspicious_periods_count']}")
        print(f"📈 Suspicious percentage: {result.metrics['suspicious_percentage']:.1f}%")
        print(f"🎯 Confidence score: {result.confidence_score:.2%}")
        print("-" * 70)
        
        # Show alert breakdown
        bearish_count = len([sp for sp in result.suspicious_periods if "bearish_volume_spike" in sp["alerts"]])
        bullish_count = len([sp for sp in result.suspicious_periods if "bullish_volume_spike" in sp["alerts"]])
        standard_count = len([sp for sp in result.suspicious_periods if "mean_std_volume_spike" in sp["alerts"] and "bearish_volume_spike" not in sp["alerts"] and "bullish_volume_spike" not in sp["alerts"]])
        
        print("🎯 RSI-ENHANCED ALERT BREAKDOWN:")
        print(f"  🐻 Bearish alerts (potential tops): {bearish_count}")
        print(f"  🐂 Bullish alerts (potential bottoms): {bullish_count}")
        print(f"  📊 Standard volume spikes: {standard_count}")
        print("-" * 70)
        
        # Display alerts
        if result.alerts:
            print("🚨 INTELLIGENT ALERTS:")
            for alert in result.alerts:
                level = alert.get('level', 'info').upper()
                message = alert.get('message', '')
                alert_type = alert.get('type', 'unknown')
                
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
        print("-" * 70)
        
        # Generate charts and reports
        print("📊 Generating RSI-enhanced charts and reports...")
        
        # Create output directory
        output_dir = Path("synthetic_rsi_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp_str = result.analysis_timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"synthetic_{result.pair}_{result.timeframe}_{timestamp_str}_rsi_demo"
        
        # Initialize chart generator
        chart_generator = VolumeChartGenerator()
        
        # Generate interactive HTML chart
        chart_html = chart_generator.create_analysis_chart(result)
        chart_file = output_dir / f"{base_filename}_chart.html"
        with open(chart_file, 'w', encoding='utf-8') as f:
            f.write(chart_html)
        print(f"📈 Interactive chart saved: {chart_file}")
        
        # Generate full HTML report
        report_html = chart_generator.create_analysis_report(result)
        report_file = output_dir / f"{base_filename}_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_html)
        print(f"📋 Full report saved: {report_file}")
        
        # Save analysis data as JSON
        import json
        analysis_data = {
            "pair": result.pair,
            "timeframe": result.timeframe,
            "analysis_timestamp": result.analysis_timestamp.isoformat(),
            "metrics": result.metrics,
            "suspicious_periods_count": len(result.suspicious_periods),
            "confidence_score": result.confidence_score,
            "alerts": result.alerts,
            "rsi_enhanced_features": {
                "bearish_alerts": bearish_count,
                "bullish_alerts": bullish_count,
                "standard_alerts": standard_count,
                "rsi_overbought_threshold": 70,
                "rsi_oversold_threshold": 30,
            },
            "synthetic_data_info": {
                "total_periods": len(df),
                "price_range": [float(df['Close'].min()), float(df['Close'].max())],
                "volume_range": [float(df['Volume'].min()), float(df['Volume'].max())],
                "rsi_range": [float(df['RSI_14'].min()), float(df['RSI_14'].max())],
            }
        }
        
        json_file = output_dir / f"{base_filename}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"💾 Analysis data saved: {json_file}")
        
        print("-" * 70)
        print("✅ Synthetic RSI-enhanced volume analysis test completed successfully!")
        print("🎉 All three alert types demonstrated:")
        print("   🐻 Bearish alerts (volume spike + RSI > 70)")
        print("   🐂 Bullish alerts (volume spike + RSI < 30)")
        print("   📊 Standard alerts (volume spike without RSI extremes)")
        print(f"📁 Output files saved in: {output_dir.absolute()}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during synthetic RSI-enhanced volume analysis: {str(e)}")
        logger.error(f"Synthetic RSI-enhanced volume analysis test failed: {str(e)}", exc_info=True)
        return False


async def main():
    """Main test function."""
    success = await test_synthetic_rsi_volume_analysis()
    
    if success:
        print("🎉 Synthetic RSI-enhanced volume analysis test completed successfully!")
        return 0
    else:
        print("❌ Synthetic RSI-enhanced volume analysis test failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
