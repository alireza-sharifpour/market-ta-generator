#!/usr/bin/env python3
"""
Test script for volume analysis functionality in market-ta-generator.
This script demonstrates how to use the integrated volume analysis features.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

from app.core.volume_analyzer import VolumeAnalyzer
from app.core.volume_chart_generator import VolumeChartGenerator
from app.core.batch_volume_analyzer import BatchVolumeAnalyzer
from app.utils.logging_config import setup_logging

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


async def test_volume_analysis():
    """Test the volume analysis functionality."""
    
    print("=" * 60)
    print("Market TA Generator - Volume Analysis Test")
    print("=" * 60)
    
    # Test parameters
    pair = "btc_usdt"
    timeframe = "hour1"
    periods = 168  # 1 week of hourly data
    
    print(f"Testing volume analysis for {pair.upper()}")
    print(f"Timeframe: {timeframe}")
    print(f"Periods: {periods}")
    print(f"Method: Mean + 4×Standard Deviation")
    print("-" * 60)
    
    try:
        # Initialize analyzer and chart generator
        analyzer = VolumeAnalyzer()
        chart_generator = VolumeChartGenerator()
        
        print("Fetching data and performing analysis...")
        
        # Perform analysis
        result = await analyzer.analyze_pair(pair, timeframe, periods)
        
        print(f"Analysis completed successfully!")
        print(f"Analysis timestamp: {result.analysis_timestamp}")
        print(f"Data points: {len(result.data)}")
        print(f"Suspicious periods: {len(result.suspicious_periods)}")
        print(f"Confidence score: {result.confidence_score:.2%}")
        print("-" * 60)
        
        # Display metrics
        print("ANALYSIS METRICS:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("-" * 60)
        
        # Display alerts
        if result.alerts:
            print("ALERTS:")
            for alert in result.alerts:
                level = alert.get('level', 'info').upper()
                message = alert.get('message', '')
                print(f"  [{level}] {message}")
        else:
            print("No alerts generated.")
        print("-" * 60)
        
        # Display suspicious periods
        if result.suspicious_periods:
            print("SUSPICIOUS PERIODS:")
            for i, period in enumerate(result.suspicious_periods[:5]):  # Show first 5
                timestamp = period['timestamp']
                score = period.get('score', 0)
                volume = period.get('volume', 0)
                spike_ratio = period.get('volume_spike_ratio', 1.0)
                print(f"  {i+1}. {timestamp} - Score: {score}, Volume: {volume:,.0f}, Ratio: {spike_ratio:.1f}x")
            
            if len(result.suspicious_periods) > 5:
                print(f"  ... and {len(result.suspicious_periods) - 5} more periods")
        else:
            print("No suspicious periods detected.")
        print("-" * 60)
        
        # Generate outputs
        print("Generating charts and reports...")
        
        # Create output directory
        output_dir = Path("volume_analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{pair}_{timeframe}_{timestamp_str}"
        
        # Generate interactive HTML chart
        chart_html = chart_generator.create_analysis_chart(result)
        chart_file = output_dir / f"{base_filename}_chart.html"
        with open(chart_file, 'w', encoding='utf-8') as f:
            f.write(chart_html)
        print(f"Interactive chart saved: {chart_file}")
        
        # Generate full HTML report
        report_html = chart_generator.create_analysis_report(result)
        report_file = output_dir / f"{base_filename}_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_html)
        print(f"Full report saved: {report_file}")
        
        # Save analysis data as JSON
        analysis_data = {
            "pair": result.pair,
            "timeframe": result.timeframe,
            "analysis_timestamp": result.analysis_timestamp.isoformat(),
            "metrics": result.metrics,
            "suspicious_periods_count": len(result.suspicious_periods),
            "confidence_score": result.confidence_score,
            "alerts": result.alerts
        }
        
        json_file = output_dir / f"{base_filename}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        print(f"Analysis data saved: {json_file}")
        
        print("-" * 60)
        print("✅ Volume analysis test completed successfully!")
        print(f"📁 Output files saved in: {output_dir.absolute()}")
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during volume analysis: {str(e)}")
        logger.error(f"Volume analysis test failed: {str(e)}", exc_info=True)
        return False


async def test_batch_analysis():
    """Test batch volume analysis with a small subset of pairs."""
    
    print("\n" + "=" * 60)
    print("Batch Analysis Test")
    print("=" * 60)
    
    try:
        # Create a test configuration with just a few pairs
        test_config = {
            "pairs": ["btc_usdt", "eth_usdt", "sol_usdt"],  # Test with 3 pairs
            "timeframe": "hour1",
            "periods": 24,  # 1 day of hourly data
            "max_concurrent": 2,
            "output_dir": "./test_batch_output",
            "retry_attempts": 2,
            "retry_delay": 1,
        }
        
        print(f"Testing batch analysis with {len(test_config['pairs'])} pairs")
        print(f"Pairs: {', '.join(test_config['pairs'])}")
        print(f"Timeframe: {test_config['timeframe']}")
        print("-" * 60)
        
        # Initialize batch analyzer
        analyzer = BatchVolumeAnalyzer(test_config)
        
        # Run analysis
        summary = await analyzer.analyze_all_pairs()
        
        # Display results
        batch_summary = summary["batch_analysis_summary"]
        print(f"Batch Analysis Results:")
        print(f"  - Total pairs: {batch_summary['total_pairs']}")
        print(f"  - Successful: {batch_summary['successful_pairs']}")
        print(f"  - Failed: {batch_summary['failed_pairs']}")
        print(f"  - Suspicious: {batch_summary['pairs_with_suspicious_periods']}")
        print(f"  - Duration: {batch_summary['duration_seconds']:.2f}s")
        print(f"  - Success rate: {batch_summary['success_rate']:.1f}%")
        
        # Show top suspicious pairs
        top_pairs = summary["top_suspicious_pairs"]
        if top_pairs:
            print(f"Top suspicious pairs:")
            for i, pair in enumerate(top_pairs, 1):
                print(f"  {i}. {pair['pair'].upper()}: {pair['suspicious_periods']} periods (confidence: {pair['confidence_score']:.2%})")
        
        print("✅ Batch analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Batch analysis test failed: {str(e)}")
        logger.error(f"Batch analysis test failed: {str(e)}", exc_info=True)
        return False


def print_usage_instructions():
    """Print instructions for using the volume analysis service."""
    
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("1. SETUP CRON JOB (Recommended):")
    print("   cd /path/to/market-ta-generator")
    print("   ./scripts/setup_volume_analysis_cron.sh")
    print()
    
    print("2. SETUP SYSTEMD SERVICE (Alternative):")
    print("   cd /path/to/market-ta-generator")
    print("   sudo ./scripts/setup_volume_analysis_systemd.sh")
    print()
    
    print("3. MANUAL EXECUTION:")
    print("   cd /path/to/market-ta-generator")
    print("   source venv/bin/activate")
    print("   python volume_analysis_cron.py")
    print()
    
    print("4. MONITORING:")
    print("   # View cron logs")
    print("   tail -f logs/volume_analysis_cron.log")
    print()
    print("   # View systemd logs")
    print("   sudo journalctl -u volume-analysis.service -f")
    print()
    
    print("5. OUTPUT FILES:")
    print("   - Location: ./volume_analysis_results/")
    print("   - JSON data: {pair}_{timeframe}_{timestamp}_data.json")
    print("   - HTML charts: {pair}_{timeframe}_{timestamp}_chart.html")
    print("   - HTML reports: {pair}_{timeframe}_{timestamp}_report.html")
    print("   - Batch summary: batch_summary_{timestamp}.json")
    print()
    
    print("=" * 60)


async def main():
    """Main test function."""
    
    print("🚀 Starting Market TA Generator Volume Analysis Tests")
    
    # Test 1: Core volume analysis
    success1 = await test_volume_analysis()
    
    # Test 2: Batch analysis
    success2 = await test_batch_analysis()
    
    # Print usage instructions
    print_usage_instructions()
    
    if success1 and success2:
        print("🎉 All tests completed successfully!")
        print("✅ Volume analysis integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
