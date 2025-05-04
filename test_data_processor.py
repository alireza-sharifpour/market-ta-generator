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

        logger.info("Script execution finished successfully.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Script execution failed: {e}")
        sys.exit(1)
