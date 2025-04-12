#!/usr/bin/env python3
"""
Simple test script to verify the LBank client's fetch_ohlcv function works correctly.
"""
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from urllib.parse import urljoin

import requests

from app.config import (
    LBANK_API_BASE_URL,
    LBANK_API_KEY,
    LBANK_API_SECRET,
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


def _create_signature(params, secret_key):
    """Create signature for LBank API authentication"""
    # Sort parameters by key alphabetically
    sorted_params = sorted(params.items())

    # Create parameter string
    param_str = "&".join([f"{k}={v}" for k, v in sorted_params])

    # Create signature using HMAC-SHA256
    signature = hmac.new(
        secret_key.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return signature


def test_direct_api():
    """Test the LBank API directly to debug the candle count issue."""
    url = urljoin(LBANK_API_BASE_URL, LBANK_KLINE_ENDPOINT)

    # Create parameters
    params = {
        "symbol": "eth_usdt",
        "size": 20,  # Explicitly request 20 candles
        "type": "hour1",
        "time": int(time.time()),
    }

    # Add authentication if available
    if LBANK_API_KEY and LBANK_API_SECRET:
        print(f"Using API key: {LBANK_API_KEY[:5]}... and secret for authentication")
        params["api_key"] = LBANK_API_KEY
        params["sign"] = _create_signature(params, LBANK_API_SECRET)
    else:
        print("No API credentials available")

    print(f"Making direct API request to: {url}")
    # print(f"With parameters: {params}")

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"API Status code: {response.status_code}")
        print(f"API Response: {response.text[:500]}...")  # Show first 500 chars

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                print(f"Number of candles: {len(data['data'])}")
            else:
                print(
                    f"Number of candles: {len(data) if isinstance(data, list) else 'Not a list'}"
                )
    except Exception as e:
        print(f"Direct API request failed: {str(e)}")


def test_fetch_ohlcv():
    """Test the fetch_ohlcv function with a common trading pair."""
    try:
        # Test with eth_usdt pair, using hour1 timeframe and 20 data points
        # Try a bigger number explicitly
        data = fetch_ohlcv(pair="eth_usdt", timeframe="hour1", limit=20)

        print(f"Successfully fetched {len(data)} data points.")
        print(f"Raw response data: {data}")

        # Print the first data point as an example
        if data and len(data) > 0:
            print("\nExample data point:")
            point = data[0]
            print(f"Timestamp: {point[0]}")
            print(f"Open: {point[1]}")
            print(f"High: {point[2]}")
            print(f"Low: {point[3]}")
            print(f"Close: {point[4]}")
            print(f"Volume: {point[5]}")

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
