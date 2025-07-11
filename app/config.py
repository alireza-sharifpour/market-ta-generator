import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LBank API Configuration
LBANK_API_KEY = os.getenv("LBANK_API_KEY", "")
LBANK_API_SECRET = os.getenv("LBANK_API_SECRET", "")
LBANK_API_BASE_URL = "https://api.lbkex.com"
LBANK_KLINE_ENDPOINT = "/v2/kline.do"

# Avalai API Configuration
AVALAI_API_KEY = os.getenv("AVALAI_API_KEY", "")
AVALAI_API_BASE_URL = "https://api.avalai.ir/v1"

# Default parameters for OHLCV data fetching
DEFAULT_TIMEFRAME = "day1"
DEFAULT_SIZE = 200

# Technical Indicators Configuration
# Default periods/settings for technical indicators
INDICATOR_SETTINGS = {
    "ema_short": 9,
    "ema_medium": 21,
    "ema_long": 50,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "adx_period": 14,
    "mfi_period": 14,
}

# Support/Resistance Detection Configuration
# Timeframe-specific parameters for support/resistance detection
SR_SETTINGS = {
    # High-frequency timeframes (1m, 5m, 15m, 30m)
    "high_frequency": {
        "timeframes": ["minute1", "minute5", "minute15", "minute30"],
        "lookback_periods": 5,
        "cluster_threshold_percent": 0.3,
        "min_touches": 2,
        "top_n_levels": 6,
        "density_window": 3,
        "volatility_adjustment": True,
    },
    # Medium-frequency timeframes (1h, 4h, 8h, 12h)
    "medium_frequency": {
        "timeframes": ["hour1", "hour4", "hour8", "hour12"],
        "lookback_periods": 8,
        "cluster_threshold_percent": 0.5,
        "min_touches": 2,
        "top_n_levels": 5,
        "density_window": 5,
        "volatility_adjustment": True,
    },
    # Low-frequency timeframes (1d, 1w, 1M)
    "low_frequency": {
        "timeframes": ["day1", "week1", "month1"],
        "lookback_periods": 12,
        "cluster_threshold_percent": 1.0,
        "min_touches": 2,
        "top_n_levels": 5,
        "density_window": 7,
        "volatility_adjustment": False,
    },
    # Default settings when timeframe is not specified
    "default": {
        "lookback_periods": 10,
        "cluster_threshold_percent": 1.0,
        "min_touches": 2,
        "top_n_levels": 5,
        "density_window": 5,
        "volatility_adjustment": True,
    },
}

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gemini-2.5-flash-preview-04-17"

# IP Whitelist Configuration
# Default values provided, but can be overridden with environment variables
WHITELIST_ENABLED = os.getenv("WHITELIST_ENABLED", "True").lower() == "true"
# Define whitelisted IPs as a comma-separated string in the environment
WHITELISTED_IPS = os.getenv(
    "WHITELISTED_IPS", "127.0.0.1,154.90.55.18,10.72.24.67,4.210.246.79"
).split(",")
