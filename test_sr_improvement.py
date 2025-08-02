#!/usr/bin/env python3
"""
Test script to verify the improved S/R detection against the user's example.
This script creates synthetic data similar to the user's chart to test
if the enhanced algorithm can detect the support level around 2525.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.core.data_processor import identify_support_resistance, process_raw_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_data():
    """
    Create synthetic OHLCV data similar to the user's chart example.
    The data will include a clear support level around 2525.
    """
    # Create base timestamps (hourly data for 200 periods)
    start_time = datetime(2025, 7, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(200)]

    # Create price data with a clear support level around 2525
    np.random.seed(42)  # For reproducible results

    base_prices = []
    current_price = 2600  # Start at 2600

    for i in range(200):
        # Add some trend and volatility
        trend = -0.2 if i < 100 else 0.5  # Downtrend then uptrend
        volatility = np.random.normal(0, 10)

        # Strong support at 2525 - price bounces when it gets close
        if current_price < 2530:
            # Strong bounce from support level
            bounce_strength = max(0, 2525 - current_price) * 0.8
            current_price += bounce_strength + abs(volatility)
        else:
            current_price += trend + volatility

        # Add some random variation but keep it realistic
        current_price = max(2500, min(2700, current_price))
        base_prices.append(current_price)

    # Create OHLCV data
    raw_data = []
    for i, (timestamp, base_price) in enumerate(zip(timestamps, base_prices)):
        # Create realistic OHLC based on the base price
        daily_range = np.random.uniform(15, 35)  # Random daily range

        open_price = base_price + np.random.uniform(-5, 5)
        high = open_price + np.random.uniform(5, daily_range)
        low = open_price - np.random.uniform(5, daily_range)
        close = open_price + np.random.uniform(-daily_range / 2, daily_range / 2)

        # Ensure support level is tested multiple times
        if i > 20 and i < 180:  # Don't apply to very early or very late data
            # If we're near the support level, make sure we touch it
            if 2520 <= min(low, close) <= 2530:
                low = 2525 + np.random.uniform(-2, 2)  # Touch the support level
                if close < low:
                    close = low + np.random.uniform(0, 10)  # Bounce from support

        # Ensure OHLC logic is correct
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = np.random.uniform(500, 2000)

        # Convert to timestamp format expected by the processor
        timestamp_unix = int(timestamp.timestamp())

        raw_data.append([timestamp_unix, open_price, high, low, close, volume])

    return raw_data


def test_enhanced_sr_detection():
    """Test the enhanced S/R detection with synthetic data."""
    logger.info("=== Testing Enhanced S/R Detection ===")

    # Create synthetic data
    logger.info("Creating synthetic data with support level around 2525...")
    raw_data = create_synthetic_data()

    # Process the data
    logger.info("Processing raw data...")
    df = process_raw_data(raw_data)

    logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Price range: {df['Low'].min():.2f} to {df['High'].max():.2f}")
    logger.info(f"Final price: {df['Close'].iloc[-1]:.2f}")

    # Test different timeframes
    timeframes = ["minute1", "hour1", "day1"]

    for timeframe in timeframes:
        logger.info(f"\n--- Testing timeframe: {timeframe} ---")

        try:
            sr_levels = identify_support_resistance(df, timeframe=timeframe)

            logger.info("Support levels found: %s", sr_levels["support"])
            logger.info("Resistance levels found: %s", sr_levels["resistance"])

            # Check if we detected the target support level around 2525
            target_level = 2525
            tolerance = 5  # Within 5 points

            support_near_target = [
                level
                for level in sr_levels["support"]
                if abs(level - target_level) <= tolerance
            ]

            if support_near_target:
                logger.info(
                    f"✅ SUCCESS: Found support level(s) near {target_level}: {support_near_target}"
                )
            else:
                logger.info(
                    f"❌ Target level {target_level} not detected in timeframe {timeframe}"
                )

                # Show closest levels for debugging
                if sr_levels["support"]:
                    closest_support = min(
                        sr_levels["support"], key=lambda x: abs(x - target_level)
                    )
                    logger.info(
                        f"Closest support level: {closest_support} (distance: {abs(closest_support - target_level):.2f})"
                    )

        except Exception as e:
            logger.error(f"Error testing timeframe {timeframe}: {e}")

    # Also test with very sensitive parameters to see if we can detect the level
    logger.info(f"\n--- Testing with very sensitive parameters ---")
    try:
        sr_levels_sensitive = identify_support_resistance(
            df,
            lookback_periods=3,
            cluster_threshold_percent=0.1,
            min_touches=2,
            top_n_levels=10,
        )

        logger.info("Sensitive - Support levels: %s", sr_levels_sensitive["support"])
        logger.info(
            "Sensitive - Resistance levels: %s", sr_levels_sensitive["resistance"]
        )

        # Check again for the target level
        support_near_target = [
            level for level in sr_levels_sensitive["support"] if abs(level - 2525) <= 5
        ]

        if support_near_target:
            logger.info(
                f"✅ SUCCESS with sensitive params: Found support level(s) near 2525: {support_near_target}"
            )
        else:
            logger.info(
                f"❌ Target level 2525 still not detected with sensitive parameters"
            )

    except Exception as e:
        logger.error(f"Error testing with sensitive parameters: {e}")


if __name__ == "__main__":
    test_enhanced_sr_detection()
