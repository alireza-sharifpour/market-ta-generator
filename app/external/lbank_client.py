import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
import uuid
from datetime import datetime
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


def fetch_ohlcv(
    pair: str, timeframe: Optional[str] = None, limit: Optional[int] = None
) -> List[List[Union[int, float]]]:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from LBank API.

    Note: Despite what the LBank API documentation suggests, their kline endpoint
    seems to only return a single candle (the most recent one) regardless of the
    'size' parameter value. This function will still attempt to request multiple
    candles as specified, but expect to receive only one.

    Args:
        pair: Trading pair symbol (e.g., "eth_btc")
        timeframe: Time interval for each candle. Options include:
                  minute1, minute5, minute15, minute30, hour1, hour4, hour8, hour12, day1, week1, month1
        limit: Number of candles to fetch (1-2000 for v2 API)
             Note: The API currently ignores this parameter and returns only 1 candle.

    Returns:
        List of lists with format [timestamp, open, high, low, close, volume]
        Currently only contains a single candle per LBank API behavior.

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

    # Basic parameters for the request
    params: Dict[str, Union[str, int]] = {
        "symbol": pair,
        "size": limit_to_use,  # Note: API currently ignores this
        "type": timeframe_to_use,
        "time": int(time.time()),
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
            f"Response body: {response.text[:500]}..."
        )  # Truncate for log readability

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

                if data_length != limit_to_use and limit_to_use > 1:
                    logger.warning(
                        f"LBank API returned {data_length} candles even though {limit_to_use} were requested. "
                        "This appears to be an API limitation."
                    )

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

                return data
            else:
                error_msg = "LBank API response does not contain data field"
                logger.error(error_msg)
                raise LBankAPIError(error_msg)

        # Fallback for v1 API format (directly returns list of candles)
        elif isinstance(response_data, list):
            data = response_data
            data_length = len(data)

            if data_length != limit_to_use and limit_to_use > 1:
                logger.warning(
                    f"LBank API returned {data_length} candles even though {limit_to_use} were requested. "
                    "This appears to be an API limitation."
                )

            logger.info(
                f"Successfully fetched {data_length} OHLCV entries for {pair} (v1 API)"
            )

            # Validate each data point has the expected structure
            for candle in data:
                if not isinstance(candle, list) or len(candle) < 6:
                    error_msg = f"Invalid candle data format: {candle}"
                    logger.error(error_msg)
                    raise LBankAPIError(error_msg)

            return data

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
