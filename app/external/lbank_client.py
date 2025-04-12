import hashlib
import hmac
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from app.config import (
    DEFAULT_SIZE,
    DEFAULT_TIMEFRAME,
    LBANK_API_BASE_URL,
    LBANK_API_KEY,
    LBANK_API_SECRET,
    LBANK_KLINE_ENDPOINT,
)

# Set up logging
logger = logging.getLogger(__name__)


class LBankAPIError(Exception):
    """Exception raised for LBank API errors"""

    pass


class LBankConnectionError(Exception):
    """Exception raised for connection errors with LBank API"""

    pass


def _create_signature(params: Dict[str, Any], secret_key: str) -> str:
    """
    Create signature for LBank API authentication using HMAC-SHA256 method.

    Args:
        params: Request parameters (excluding 'sign')
        secret_key: API secret key

    Returns:
        Signature string
    """
    # Sort parameters by key alphabetically
    sorted_params = sorted(params.items())

    # Create parameter string
    param_str = "&".join([f"{k}={v}" for k, v in sorted_params])

    # Create MD5 digest of parameters (uppercase)
    md5_digest = hashlib.md5(param_str.encode("utf-8")).hexdigest().upper()

    # Create signature using HMAC-SHA256
    signature = hmac.new(
        secret_key.encode("utf-8"), md5_digest.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return signature


def _calculate_past_timestamp(timeframe: str, periods: int = 50) -> int:
    """
    Calculate a timestamp in the past based on the timeframe and number of periods.
    This helps us retrieve historical data from LBank API.

    Args:
        timeframe: The timeframe of candles (e.g., "hour1", "day1")
        periods: Number of periods to go back

    Returns:
        UNIX timestamp (seconds) for the calculated past time
    """
    current_time = time.time()
    seconds_to_subtract = 0

    # Map timeframe to seconds
    if timeframe == "minute1":
        seconds_to_subtract = 60 * periods
    elif timeframe == "minute5":
        seconds_to_subtract = 300 * periods
    elif timeframe == "minute15":
        seconds_to_subtract = 900 * periods
    elif timeframe == "minute30":
        seconds_to_subtract = 1800 * periods
    elif timeframe == "hour1":
        seconds_to_subtract = 3600 * periods
    elif timeframe == "hour4":
        seconds_to_subtract = 14400 * periods
    elif timeframe == "hour8":
        seconds_to_subtract = 28800 * periods
    elif timeframe == "hour12":
        seconds_to_subtract = 43200 * periods
    elif timeframe == "day1":
        seconds_to_subtract = 86400 * periods
    elif timeframe == "week1":
        seconds_to_subtract = 604800 * periods
    elif timeframe == "month1":
        seconds_to_subtract = 2592000 * periods
    else:
        # Default to 2 days if timeframe is unknown
        seconds_to_subtract = 172800

    # Calculate past timestamp
    past_timestamp = int(current_time - seconds_to_subtract)
    return past_timestamp


def fetch_ohlcv(
    pair: str, timeframe: Optional[str] = None, limit: Optional[int] = None
) -> List[List[Union[int, float]]]:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from LBank API.

    LBank API requires using a past timestamp to retrieve historical data. This function
    automatically calculates an appropriate past timestamp based on the timeframe
    and requested limit to ensure multiple candles are returned.

    Args:
        pair: Trading pair symbol (e.g., "eth_btc")
        timeframe: Time interval for each candle. Options include:
                  minute1, minute5, minute15, minute30, hour1, hour4, hour8, hour12, day1, week1, month1
        limit: Number of candles to fetch (1-2000 for v2 API)

    Returns:
        List of lists with format [timestamp, open, high, low, close, volume]

    Raises:
        LBankAPIError: If the LBank API returns an error
        LBankConnectionError: If there's a connection issue
        ValueError: If input parameters are invalid
    """
    # Use default values if not provided
    timeframe_to_use: str = DEFAULT_TIMEFRAME if timeframe is None else timeframe
    limit_to_use: int = DEFAULT_SIZE if limit is None else limit

    # Validate inputs
    if not pair:
        raise ValueError("Trading pair is required")
    if not isinstance(limit_to_use, int) or limit_to_use < 1 or limit_to_use > 2000:
        raise ValueError("Limit must be an integer between 1 and 2000")

    # Valid timeframe values according to LBank API
    valid_timeframes = [
        "minute1",
        "minute5",
        "minute15",
        "minute30",
        "hour1",
        "hour4",
        "hour8",
        "hour12",
        "day1",
        "week1",
        "month1",
    ]

    if timeframe_to_use not in valid_timeframes:
        raise ValueError(
            f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
        )

    # Construct API URL and parameters
    url = urljoin(LBANK_API_BASE_URL, LBANK_KLINE_ENDPOINT)

    # Calculate past timestamp to get historical data
    # We no longer use a buffer multiplier - instead, we go back exactly 'limit' periods
    # This makes the behavior more intuitive - if you request 10 daily candles, you get data from 10 days ago
    past_timestamp = _calculate_past_timestamp(timeframe_to_use, limit_to_use)
    logger.info(
        f"Using past timestamp {past_timestamp} to fetch historical data (exactly {limit_to_use} {timeframe_to_use} periods ago)"
    )

    # Basic parameters for the request
    params: Dict[str, Union[str, int]] = {
        "symbol": pair,
        "size": limit_to_use,
        "type": timeframe_to_use,
        "time": past_timestamp,  # Using past timestamp to get historical data
    }

    # Set up authentication headers and parameters
    headers: Dict[str, str] = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # Handle authentication according to LBank's documentation
    if LBANK_API_KEY and LBANK_API_SECRET:
        logger.info("Using LBank API authentication")

        # Add required authentication parameters
        auth_params = {
            **params,  # Include the existing parameters
            "api_key": LBANK_API_KEY,
            "timestamp": int(time.time() * 1000),  # Milliseconds as required by LBank
            "signature_method": "HmacSHA256",
            "echostr": str(uuid.uuid4())[:35],  # Random string between 30-40 chars
        }

        # Generate signature
        sign = _create_signature(auth_params, LBANK_API_SECRET)

        # Add the signature to the parameters
        params = {**auth_params, "sign": sign}

        logger.info("Authentication parameters added to request")
    else:
        logger.warning(
            "API key or secret missing. Using unauthenticated request (public endpoints only)."
        )

    # Log detailed request information for debugging
    logger.info(f"Making request to URL: {url}")
    # logger.info(f"With parameters: {params}")
    logger.info(f"Headers: {headers}")

    try:
        # Make the API request
        response = requests.get(url, params=params, headers=headers, timeout=30)

        # Check if the request was successful
        if response.status_code != 200:
            error_msg = f"LBank API returned status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise LBankAPIError(error_msg)

        # Log full response for debugging
        logger.info(f"Response status code: {response.status_code}")
        logger.info(
            f"Response body length: {len(response.text)}"
        )  # Only log the length to avoid huge logs

        # Parse the response
        response_data = response.json()

        # V2 API error handling
        if isinstance(response_data, dict):
            # Check for error in the v2 API format
            if (
                response_data.get("error_code", "00000") != 0
                and response_data.get("error_code", "00000") != "00000"
            ):
                error_msg = f"LBank API error: {response_data.get('error_code', 'Unknown')}, {response_data.get('msg', '')}"
                logger.error(error_msg)
                raise LBankAPIError(error_msg)

            # For v2 API, extract the actual data from the 'data' field
            if "data" in response_data:
                data = response_data["data"]
                data_length = len(data) if isinstance(data, list) else 0

                logger.info(
                    f"Successfully fetched {data_length} OHLCV entries for {pair} (v2 API)"
                )

                # Validate that the data is a list
                if not isinstance(data, list):
                    error_msg = f"Unexpected data format from LBank API: {data}"
                    logger.error(error_msg)
                    raise LBankAPIError(error_msg)

                # Validate each data point has the expected structure
                for candle in data:
                    if not isinstance(candle, list) or len(candle) < 6:
                        error_msg = f"Invalid candle data format: {candle}"
                        logger.error(error_msg)
                        raise LBankAPIError(error_msg)

                # Sort the data by timestamp in chronological order (oldest to newest)
                sorted_data = sorted(data, key=lambda x: x[0])
                result = sorted_data[:limit_to_use]

                logger.info(
                    f"Returning {len(result)} candles in chronological order (oldest first)"
                )
                return result
            else:
                error_msg = "LBank API response does not contain data field"
                logger.error(error_msg)
                raise LBankAPIError(error_msg)

        # Fallback for v1 API format (directly returns list of candles)
        elif isinstance(response_data, list):
            data = response_data
            data_length = len(data)

            logger.info(
                f"Successfully fetched {data_length} OHLCV entries for {pair} (v1 API)"
            )

            # Validate each data point has the expected structure
            for candle in data:
                if not isinstance(candle, list) or len(candle) < 6:
                    error_msg = f"Invalid candle data format: {candle}"
                    logger.error(error_msg)
                    raise LBankAPIError(error_msg)

            # Sort the data by timestamp in chronological order (oldest to newest)
            sorted_data = sorted(data, key=lambda x: x[0])
            result = sorted_data[:limit_to_use]

            logger.info(
                f"Returning {len(result)} candles in chronological order (oldest first)"
            )
            return result

        else:
            error_msg = f"Unexpected response format from LBank API: {response_data}"
            logger.error(error_msg)
            raise LBankAPIError(error_msg)

    except requests.exceptions.RequestException as e:
        error_msg = f"Connection error when querying LBank API: {str(e)}"
        logger.error(error_msg)
        raise LBankConnectionError(error_msg)
    except (ValueError, KeyError, TypeError) as e:
        error_msg = f"Error parsing response from LBank API: {str(e)}"
        logger.error(error_msg)
        raise LBankAPIError(error_msg)
