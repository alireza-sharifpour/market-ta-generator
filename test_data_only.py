#!/usr/bin/env python
"""
Test script for testing just the data fetching and processing parts
of the market-ta-generator analysis service.
"""

import json
import logging
import sys

from app.core.data_processor import format_data_for_llm, process_raw_data
from app.external.lbank_client import fetch_ohlcv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_pipeline(pair):
    """
    Test the data fetching and processing pipeline without the LLM step.

    Args:
        pair: Trading pair to analyze (e.g., "eth_usdt")
    """
    logger.info(f"Testing data pipeline for pair: {pair}")

    try:
        # Step 1: Fetch raw OHLCV data from LBank API
        logger.info(f"Fetching OHLCV data for {pair}")
        raw_data = fetch_ohlcv(pair)
        logger.info(f"Successfully fetched {len(raw_data)} data points")

        # Print a sample of the raw data
        logger.info(f"Sample of raw data (first 3 entries):")
        for i, entry in enumerate(raw_data[:3]):
            logger.info(f"  Entry {i}: {entry}")

        # Step 2: Process the raw data into a DataFrame
        logger.info("Processing raw data into DataFrame")
        df = process_raw_data(raw_data)
        logger.info(f"Successfully processed data into DataFrame with {len(df)} rows")

        # Print DataFrame info and sample
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info("DataFrame sample (first 3 rows):")
        logger.info(df.head(3))

        # Step 3: Format the data for LLM consumption
        logger.info("Formatting data for LLM")
        formatted_data = format_data_for_llm(df)

        # Print a portion of the formatted data
        logger.info("Sample of formatted data (first 300 chars):")
        logger.info(formatted_data[:300] + "...")

        logger.info("Data pipeline test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        return False


if __name__ == "__main__":
    # Use command line argument or default to "eth_usdt"
    pair = sys.argv[1] if len(sys.argv) > 1 else "eth_usdt"
    test_data_pipeline(pair)
