#!/usr/bin/env python3
"""
Test script to verify that the data_processor module functions correctly.
"""

import logging
import os
import sys
import time
from datetime import datetime

from app.core.data_processor import format_data_for_llm, process_raw_data
from app.external.lbank_client import fetch_ohlcv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our modules


def test_data_processor():
    """Test the data processor with real data from LBank."""
    try:
        # Fetch some sample data from LBank
        pair = "eth_usdt"
        timeframe = "day1"
        limit = 50

        print(
            f"Fetching OHLCV data for {pair} with timeframe={timeframe}, limit={limit}"
        )
        raw_data = fetch_ohlcv(pair=pair, timeframe=timeframe, limit=limit)
        print(f"Successfully fetched {len(raw_data)} data points.")

        # Process the raw data
        print("\nProcessing raw data into DataFrame...")
        df = process_raw_data(raw_data)
        print(f"Processed DataFrame shape: {df.shape}")

        # Display the first few rows of the DataFrame
        print("\nFirst 3 rows of the processed DataFrame:")
        print(df.head(3))

        # Format the data for LLM
        print("\nFormatting data for LLM...")
        formatted_data = format_data_for_llm(df)
        print("\nFormatted data for LLM:")
        print(formatted_data)

        return True
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing data processor functionality...")
    success = test_data_processor()

    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
