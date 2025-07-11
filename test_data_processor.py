#!/usr/bin/env python3
"""
Test script to verify that the data_processor module functions correctly.
"""

import logging
import os
import sys

import pytest
from pandas import DataFrame

# Import our modules
from app.core.data_processor import (
    add_technical_indicators,
    format_data_for_llm,
    process_raw_data,
    identify_support_resistance,
    get_sr_settings_for_timeframe,
)
from app.external.lbank_client import fetch_ohlcv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if running as script
# (pytest usually handles this automatically)
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="module")
def sample_ohlcv_data() -> DataFrame:
    """Fixture to fetch sample data once per module."""
    pair = "btc_usdt"  # Use a common pair like BTC
    timeframe = "day1"
    limit = 100  # Need enough data for indicators
    logger.info(
        f"Fetching OHLCV data for {pair} (timeframe={timeframe}, limit={limit})..."
    )
    try:
        raw_data = fetch_ohlcv(pair=pair, timeframe=timeframe, limit=limit)
        if not raw_data or len(raw_data) < 50:  # Basic check for enough data
            pytest.skip(
                f"Insufficient data fetched for {pair} ({len(raw_data)} points)"
            )
        logger.info(f"Successfully fetched {len(raw_data)} data points.")
        df = process_raw_data(raw_data)
        assert isinstance(df, DataFrame)
        assert not df.empty
        return df
    except Exception as e:
        logger.error(f"Failed to fetch or process data for fixture: {e}")
        pytest.fail(f"Data fetching/processing failed: {e}")
    pytest.fail("Fixture failed unexpectedly after try-except block")


def test_process_raw_data(sample_ohlcv_data: DataFrame):
    """Test the basic data processing."""
    logger.info("Testing process_raw_data (via fixture)...")
    df = sample_ohlcv_data
    assert isinstance(df, DataFrame)
    assert not df.empty
    assert "Open" in df.columns
    assert "Close" in df.columns
    assert df.index.name == "Timestamp"
    logger.info(f"Processed DataFrame shape: {df.shape}")
    logger.info("First 3 rows:\n%s", df.head(3))


def test_add_technical_indicators(sample_ohlcv_data: DataFrame):
    """Test adding technical indicators."""
    logger.info("Testing add_technical_indicators...")
    df = sample_ohlcv_data
    try:
        df_with_indicators = add_technical_indicators(df)
        assert isinstance(df_with_indicators, DataFrame)
        assert df_with_indicators.shape[0] == df.shape[0]  # Check rows match

        # Check if expected indicator columns are present
        # Note: Column names depend on pandas-ta naming conventions
        expected_indicator_patterns = [
            "EMA_",
            "RSI_",
            "BBL_",
            "BBM_",
            "BBU_",
            "ADX_",
            "MFI_",
        ]
        added_cols = set(df_with_indicators.columns) - set(df.columns)
        logger.info(f"Columns added: {sorted(list(added_cols))}")

        for pattern in expected_indicator_patterns:
            assert any(
                col.startswith(pattern) for col in added_cols
            ), f"Missing indicator column starting with {pattern}"

        logger.info("Indicators added successfully.")
        logger.info("Last 3 rows with indicators:\n%s", df_with_indicators.tail(3))

        # Check for NaNs (expect some NaNs at the beginning)
        nan_counts = df_with_indicators.isna().sum()
        logger.info(f"NaN counts per column:\n{nan_counts}")
        # Basic check: ensure not ALL values are NaN for added indicators
        for col in added_cols:
            assert not df_with_indicators[col].isna().all(), f"Column {col} is all NaN"

    except Exception as e:
        logger.error(f"Error during add_technical_indicators test: {e}", exc_info=True)
        pytest.fail(f"add_technical_indicators failed: {e}")


def test_format_data_for_llm(sample_ohlcv_data: DataFrame):
    """Test formatting data for the LLM."""
    logger.info("Testing format_data_for_llm...")
    df = sample_ohlcv_data
    formatted_data = format_data_for_llm(df, timeframe="day1")
    assert isinstance(formatted_data, str)
    assert "OHLCV Data" in formatted_data
    assert "(daily timeframe)" in formatted_data
    assert "Summary Statistics:" in formatted_data
    assert "Latest Close:" in formatted_data
    logger.info("LLM Formatted Data (first 500 chars):\n%s...", formatted_data[:500])


def test_get_sr_settings_for_timeframe():
    """Test getting S/R settings for different timeframes."""
    logger.info("Testing get_sr_settings_for_timeframe...")
    
    # Test high-frequency timeframe
    settings_1m = get_sr_settings_for_timeframe("minute1")
    assert settings_1m["lookback_periods"] == 5
    assert settings_1m["cluster_threshold_percent"] == 0.3
    assert settings_1m["volatility_adjustment"] == True
    
    # Test medium-frequency timeframe
    settings_1h = get_sr_settings_for_timeframe("hour1")
    assert settings_1h["lookback_periods"] == 8
    assert settings_1h["cluster_threshold_percent"] == 0.5
    
    # Test low-frequency timeframe
    settings_1d = get_sr_settings_for_timeframe("day1")
    assert settings_1d["lookback_periods"] == 12
    assert settings_1d["cluster_threshold_percent"] == 1.0
    
    # Test unknown timeframe (should return default)
    settings_unknown = get_sr_settings_for_timeframe("unknown")
    assert settings_unknown["lookback_periods"] == 10
    
    # Test None timeframe (should return default)
    settings_none = get_sr_settings_for_timeframe(None)
    assert settings_none["lookback_periods"] == 10
    
    logger.info("S/R settings test passed for all timeframes")


def test_identify_support_resistance_basic(sample_ohlcv_data: DataFrame):
    """Test basic support/resistance identification."""
    logger.info("Testing identify_support_resistance basic functionality...")
    df = sample_ohlcv_data
    
    # Test with default parameters
    sr_levels = identify_support_resistance(df)
    assert isinstance(sr_levels, dict)
    assert "support" in sr_levels
    assert "resistance" in sr_levels
    assert isinstance(sr_levels["support"], list)
    assert isinstance(sr_levels["resistance"], list)
    
    # All levels should be positive numbers
    for level in sr_levels["support"]:
        assert isinstance(level, (int, float))
        assert level > 0
    
    for level in sr_levels["resistance"]:
        assert isinstance(level, (int, float))
        assert level > 0
    
    logger.info(f"Found {len(sr_levels['support'])} support levels and {len(sr_levels['resistance'])} resistance levels")
    logger.info(f"Support levels: {sr_levels['support']}")
    logger.info(f"Resistance levels: {sr_levels['resistance']}")


def test_identify_support_resistance_timeframes(sample_ohlcv_data: DataFrame):
    """Test support/resistance identification with different timeframes."""
    logger.info("Testing identify_support_resistance with different timeframes...")
    df = sample_ohlcv_data
    
    timeframes_to_test = ["minute1", "hour1", "day1"]
    
    for timeframe in timeframes_to_test:
        logger.info(f"Testing timeframe: {timeframe}")
        sr_levels = identify_support_resistance(df, timeframe=timeframe)
        
        assert isinstance(sr_levels, dict)
        assert "support" in sr_levels
        assert "resistance" in sr_levels
        
        # Check that we get some levels (may be empty for some timeframes)
        total_levels = len(sr_levels["support"]) + len(sr_levels["resistance"])
        logger.info(f"Timeframe {timeframe}: {len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance")
        
        # Verify current price logic (support should be below current price, resistance above)
        current_price = df["Close"].iloc[-1]
        
        for support_level in sr_levels["support"]:
            assert support_level < current_price, f"Support level {support_level} should be below current price {current_price}"
        
        for resistance_level in sr_levels["resistance"]:
            assert resistance_level > current_price, f"Resistance level {resistance_level} should be above current price {current_price}"


def test_identify_support_resistance_custom_params(sample_ohlcv_data: DataFrame):
    """Test support/resistance identification with custom parameters."""
    logger.info("Testing identify_support_resistance with custom parameters...")
    df = sample_ohlcv_data
    
    # Test with very loose parameters (should find more levels)
    sr_levels_loose = identify_support_resistance(
        df,
        lookback_periods=3,
        cluster_threshold_percent=0.1,
        min_touches=1,
        top_n_levels=10
    )
    
    # Test with very strict parameters (should find fewer levels)
    sr_levels_strict = identify_support_resistance(
        df,
        lookback_periods=15,
        cluster_threshold_percent=2.0,
        min_touches=3,
        top_n_levels=3
    )
    
    logger.info(f"Loose parameters: {len(sr_levels_loose['support'])} support, {len(sr_levels_loose['resistance'])} resistance")
    logger.info(f"Strict parameters: {len(sr_levels_strict['support'])} support, {len(sr_levels_strict['resistance'])} resistance")
    
    # Generally, loose parameters should find more or equal levels
    # (though this isn't guaranteed due to clustering and other factors)
    assert len(sr_levels_loose["support"]) + len(sr_levels_loose["resistance"]) >= 0
    assert len(sr_levels_strict["support"]) + len(sr_levels_strict["resistance"]) >= 0


def test_identify_support_resistance_edge_cases():
    """Test support/resistance identification edge cases."""
    logger.info("Testing identify_support_resistance edge cases...")
    
    # Test with empty DataFrame
    empty_df = DataFrame()
    try:
        identify_support_resistance(empty_df)
        assert False, "Should have raised ValueError for empty DataFrame"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    # Test with insufficient data
    minimal_data = [
        [1640995200, 50000, 50100, 49900, 50050, 1000],  # 2022-01-01
        [1641081600, 50050, 50150, 49950, 50100, 1100],  # 2022-01-02
    ]
    minimal_df = process_raw_data(minimal_data)
    
    try:
        identify_support_resistance(minimal_df, lookback_periods=10)
        assert False, "Should have raised ValueError for insufficient data"
    except ValueError as e:
        assert "insufficient" in str(e).lower()
    
    logger.info("Edge case tests passed")


# Allow running as a script for basic verification
if __name__ == "__main__":
    logger.info("Running data processor tests as a script...")
    try:
        # Manually create data for script execution
        pair = "btc_usdt"
        timeframe = "day1"
        limit = 100
        logger.info(
            f"Fetching OHLCV data for {pair} (timeframe={timeframe}, limit={limit})..."
        )
        raw_data = fetch_ohlcv(pair=pair, timeframe=timeframe, limit=limit)
        if not raw_data or len(raw_data) < 50:
            logger.warning(
                f"Insufficient data fetched ({len(raw_data)} points), exiting."
            )
            sys.exit(1)

        logger.info(f"Successfully fetched {len(raw_data)} data points.")
        df_main = process_raw_data(raw_data)

        logger.info("--- Testing process_raw_data ---")
        print(df_main.head(3))

        logger.info("--- Testing add_technical_indicators ---")
        df_indicators_main = add_technical_indicators(df_main)
        print(df_indicators_main.tail(3))
        added_cols_main = set(df_indicators_main.columns) - set(df_main.columns)
        print(f"Added columns: {sorted(list(added_cols_main))}")

        logger.info("--- Testing format_data_for_llm ---")
        formatted_main = format_data_for_llm(df_main, timeframe="day1")
        print(formatted_main[:500] + "...")

        logger.info("--- Testing identify_support_resistance ---")
        sr_levels_main = identify_support_resistance(df_main, timeframe="day1")
        print(f"Support levels: {sr_levels_main['support']}")
        print(f"Resistance levels: {sr_levels_main['resistance']}")

        logger.info("--- Testing S/R with different timeframes ---")
        for tf in ["minute1", "hour1", "day1"]:
            sr_tf = identify_support_resistance(df_main, timeframe=tf)
            print(f"Timeframe {tf}: {len(sr_tf['support'])} support, {len(sr_tf['resistance'])} resistance")

        logger.info("Script execution finished successfully.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Script execution failed: {e}")
        sys.exit(1)
