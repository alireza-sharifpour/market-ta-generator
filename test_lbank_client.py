#!/usr/bin/env python3
"""
Simple test script to verify the LBank client's fetch_ohlcv function works correctly.
"""

import logging
import os
import sys
import time
from urllib.parse import urljoin

import requests

from app.config import (
    LBANK_API_BASE_URL,
    LBANK_KLINE_ENDPOINT,
)
from app.external.lbank_client import LBankAPIError, LBankConnectionError, fetch_ohlcv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# We're not actually using dotenv in this test since we don't need API keys for public endpoints
# But we'll still import it to check that it's available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, continuing without loading .env file")


def test_direct_api():
    """Test the LBank API directly to debug the candle count issue."""
    url = urljoin(LBANK_API_BASE_URL, LBANK_KLINE_ENDPOINT)

    # You can change these parameters to test different configurations
    # Available timeframes: minute1, minute5, minute15, minute30, hour1, hour4, hour8, hour12, day1, week1, month1
    # Symbol: Any trading pair supported by LBank (e.g., "eth_usdt", "btc_usdt")
    # Size: Number of candles to request (1-2000)
    symbol = "eth_usdt"
    size = 10  # Request 10 candles
    timeframe = "day1"  # Default to daily candles

    # Calculate a past timestamp dynamically based on timeframe
    # Similar to how it's done in lbank_client.py
    current_time = int(time.time())
    seconds_to_subtract = 0

    # No longer using a buffer multiplier - going back exactly 'size' periods
    # This matches the updated logic in lbank_client.py for consistency
    requested_periods = size

    print(f"Looking back exactly {requested_periods} {timeframe} periods")

    # Map timeframe to seconds, just like in lbank_client.py
    if timeframe == "minute1":
        seconds_to_subtract = 60 * requested_periods
    elif timeframe == "minute5":
        seconds_to_subtract = 300 * requested_periods
    elif timeframe == "minute15":
        seconds_to_subtract = 900 * requested_periods
    elif timeframe == "minute30":
        seconds_to_subtract = 1800 * requested_periods
    elif timeframe == "hour1":
        seconds_to_subtract = 3600 * requested_periods
    elif timeframe == "hour4":
        seconds_to_subtract = 14400 * requested_periods
    elif timeframe == "hour8":
        seconds_to_subtract = 28800 * requested_periods
    elif timeframe == "hour12":
        seconds_to_subtract = 43200 * requested_periods
    elif timeframe == "day1":
        seconds_to_subtract = 86400 * requested_periods
    elif timeframe == "week1":
        seconds_to_subtract = 604800 * requested_periods
    elif timeframe == "month1":
        seconds_to_subtract = 2592000 * requested_periods
    else:
        # Default to 2 days if timeframe is unknown
        seconds_to_subtract = 172800

    print(
        f"Using past timestamp: {current_time - seconds_to_subtract} ({time.strftime('%Y-%m-%d', time.localtime(current_time - seconds_to_subtract))})"
    )
    print(f"Requesting {size} candles with timeframe {timeframe}")
    print(
        f"Note: This should return data from approximately {size} {timeframe} periods ago until now"
    )

    try:
        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "size": size,
                "type": timeframe,
                "time": current_time - seconds_to_subtract,
            },
            timeout=30,
        )
        print(f"API Status code: {response.status_code}")
        print(f"API Response: {response.text[:500]}...")  # Show first 500 chars

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                candles = data["data"]
                print(f"Number of candles received in raw response: {len(candles)}")

                # Sort candles by timestamp in chronological order (oldest first) to match fetch_ohlcv behavior
                sorted_candles = sorted(candles, key=lambda x: x[0])

                if sorted_candles and len(sorted_candles) > 0:
                    first_candle_timestamp = sorted_candles[0][0]
                    first_candle_date = time.strftime(
                        "%Y-%m-%d", time.localtime(first_candle_timestamp)
                    )
                    print(
                        f"First candle after sorting (oldest): {sorted_candles[0]} - Date: {first_candle_date}"
                    )

                    # Also print the newest candle for comparison
                    last_candle_timestamp = sorted_candles[-1][0]
                    last_candle_date = time.strftime(
                        "%Y-%m-%d", time.localtime(last_candle_timestamp)
                    )
                    print(
                        f"Last candle after sorting (newest): {sorted_candles[-1]} - Date: {last_candle_date}"
                    )
            else:
                candle_count = len(data) if isinstance(data, list) else 0
                print(f"Number of candles received: {candle_count}")
                if candle_count > 0:
                    # Sort candles by timestamp in chronological order (oldest first) to match fetch_ohlcv behavior
                    sorted_data = sorted(data, key=lambda x: x[0])
                    first_candle_timestamp = sorted_data[0][0]
                    first_candle_date = time.strftime(
                        "%Y-%m-%d", time.localtime(first_candle_timestamp)
                    )
                    print(
                        f"First candle after sorting (oldest): {sorted_data[0]} - Date: {first_candle_date}"
                    )
    except Exception as e:
        print(f"Direct API request failed: {str(e)}")


def test_fetch_ohlcv():
    """Test the fetch_ohlcv function with a common trading pair."""
    try:
        # Define parameters to match the direct API test for consistency
        pair = "eth_usdt"
        timeframe = "day1"
        limit = 10

        print(
            f"Testing fetch_ohlcv with pair={pair}, timeframe={timeframe}, limit={limit}"
        )
        print(
            "Note: This should match the direct API test parameters for direct comparison"
        )

        # Fetch the data using our client function
        data = fetch_ohlcv(pair=pair, timeframe=timeframe, limit=limit)

        print(f"Successfully fetched {len(data)} data points (requested {limit}).")

        # Only print a sample of the data to avoid flooding the console
        sample_data = data[:3] if len(data) > 3 else data
        print(f"Sample of response data (first 3 candles): {sample_data}")

        # Print the first data point as an example
        if data and len(data) > 0:
            print("\nExample data point:")
            point = data[0]
            point_date = time.strftime("%Y-%m-%d", time.localtime(point[0]))
            print(f"Timestamp: {point[0]} - Date: {point_date}")
            print(f"Open: {point[1]}")
            print(f"High: {point[2]}")
            print(f"Low: {point[3]}")
            print(f"Close: {point[4]}")
            print(f"Volume: {point[5]}")

            # Also print the oldest candle for comparison
            oldest_point = data[0]
            oldest_point_date = time.strftime(
                "%Y-%m-%d", time.localtime(oldest_point[0])
            )
            print(f"\nOldest candle in response:")
            print(f"Timestamp: {oldest_point[0]} - Date: {oldest_point_date}")

            # Also print the newest candle for comparison
            newest_point = data[-1]
            newest_point_date = time.strftime(
                "%Y-%m-%d", time.localtime(newest_point[0])
            )
            print(f"\nNewest candle in response:")
            print(f"Timestamp: {newest_point[0]} - Date: {newest_point_date}")

        return True

    except LBankAPIError as e:
        print(f"LBank API Error: {e}")
        return False
    except LBankConnectionError as e:
        print(f"Connection Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return False


if __name__ == "__main__":
    print("Testing LBank API directly...")
    test_direct_api()

    print("\nTesting LBank client fetch_ohlcv function...")
    success = test_fetch_ohlcv()

    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)
