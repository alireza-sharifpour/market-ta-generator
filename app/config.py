import os
from pathlib import Path

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
OPENAI_MODEL = "gemini-2.5-flash"

# IP Whitelist Configuration
# Default values provided, but can be overridden with environment variables
WHITELIST_ENABLED = os.getenv("WHITELIST_ENABLED", "True").lower() == "true"
# Define whitelisted IPs as a comma-separated string in the environment
WHITELISTED_IPS = os.getenv(
    "WHITELISTED_IPS", "127.0.0.1,154.90.55.18,10.72.24.67,4.210.246.79"
).split(",")

# Cache Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"

# Cache TTL settings by timeframe (in seconds)
# Optimized to align with candle completion cycles for maximum efficiency
CACHE_TTL_SETTINGS = {
    "minute1": 30,      # 30 seconds (50% of 1-minute candle cycle)
    "minute5": 120,     # 2 minutes (40% of 5-minute candle cycle)
    "minute15": 600,    # 10 minutes (67% of 15-minute candle cycle)
    "minute30": 1200,   # 20 minutes (67% of 30-minute candle cycle)
    "hour1": 1800,      # 30 minutes (50% of 1-hour candle cycle)
    "hour4": 7200,      # 2 hours (50% of 4-hour candle cycle)
    "hour8": 14400,     # 4 hours (50% of 8-hour candle cycle)
    "hour12": 21600,    # 6 hours (50% of 12-hour candle cycle)
    "day1": 21600,      # 6 hours (25% of 24-hour candle cycle)
    "week1": 172800,    # 2 days (29% of 7-day candle cycle)
    "month1": 604800,   # 7 days (23% of 30-day candle cycle)
    "default": 600,     # 10 minutes
}

# Cache key configuration
CACHE_KEY_PREFIX = "market_ta"
CACHE_PLACEHOLDERS = {
    "current_price": "CURRENTPRICE",
    "price_change_24h": "PRICECHANGE24H",
    "volume_24h": "VOLUME24H",
}

# S/R Reclassification Configuration
# Threshold for triggering S/R reclassification based on price movement
SR_RECLASSIFICATION_THRESHOLD = 0.005  # 0.5% price movement triggers reclassification

# Volume Analysis Configuration
VOLUME_ANALYSIS_CONFIG = {
    # Only enable the mean + std deviation method for suspicious volume detection
    "enable_volume_spike_detection": False,     # Disable traditional volume spike detection
    "enable_obv_divergence": False,             # Disable OBV divergence detection
    "enable_vwap_deviation": False,             # Disable VWAP deviation detection
    "enable_relative_volume": False,            # Disable relative volume detection
    "enable_mfi_extremes": False,               # Disable MFI extremes detection
    "enable_volume_oscillator": False,          # Disable volume oscillator detection
    
    # Mean and Standard Deviation Method (ONLY method enabled)
    "enable_mean_std_detection": True,  # Enable mean and standard deviation spike detection
    "mean_std_lookback_period": 25,     # Period for calculating mean and std (25 candles)
    
    # Three-level volume threshold system
    "volume_threshold_low_multiplier": 2.0,    # Low suspicious volume
    "volume_threshold_medium_multiplier": 4.0, # Medium suspicious volume - current default
    "volume_threshold_high_multiplier": 6.0,   # High suspicious volume
    
    # Timeframe analysis mode
    "analyze_current_timeframe_only": True,    # If True, only analyze the last timeframe; If False, analyze all timeframes
    
    # RSI Integration for Intelligent Volume Alerts
    "enable_rsi_volume_alerts": True,  # Enable RSI-enhanced volume alerts
    "rsi_period": 14,                  # RSI calculation period
    "rsi_overbought_threshold": 70,    # RSI overbought level
    "rsi_oversold_threshold": 30,      # RSI oversold level
    
    # Alert Types
    "enable_bearish_volume_alerts": True,  # Volume spike + RSI > 70 (potential top)
    "enable_bullish_volume_alerts": True,  # Volume spike + RSI < 30 (potential bottom)
    
    # Keep other settings for potential future use but with defaults
    "volume_spike_threshold": 2.5,
    "volume_extreme_threshold": 5.0,
    "volume_spike_period": 20,
    "obv_divergence_periods": 14,
    "obv_smoothing_period": 3,
    "vwap_deviation_threshold": 0.05,
    "vwap_period": 20,
    "relative_volume_lookback": 20,
    "relative_volume_threshold": 2.0,
    "volume_trend_period": 14,
    "volume_trend_threshold": 0.3,
    "min_suspicious_periods": 2,
    "confidence_threshold": 0.7,
}

# Volume Chart Configuration
VOLUME_CHART_CONFIG = {
    "width": 1200,
    "height": 800,
    "template": "plotly_white",  # Light theme consistent with market-ta-generator
    "show_volume_subplot": True,
    "show_obv_subplot": False,  # Disabled to focus on mean+std method
    "suspicious_color": "#FF6B6B",     # Red for suspicious periods
    "normal_color": "#4ECDC4",         # Teal for normal periods
    "volume_color": "#45B7D1",         # Blue for volume bars
    "obv_color": "#FFA07A",            # Light salmon for OBV line
}

# Telegram Bot Configuration
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "channel_id": os.getenv("TELEGRAM_CHANNEL_ID", ""),
    "enabled": os.getenv("TELEGRAM_ENABLED", "True").lower() == "true",
    "send_charts": True,
    "send_summary": True,
    "send_alerts_only": False,  # If True, only send when suspicious periods found
    
    # Connection timeout settings (in seconds)
    "connect_timeout": int(os.getenv("TELEGRAM_CONNECT_TIMEOUT", "30")),
    "read_timeout": int(os.getenv("TELEGRAM_READ_TIMEOUT", "60")),
    "write_timeout": int(os.getenv("TELEGRAM_WRITE_TIMEOUT", "60")),
    "connection_pool_size": int(os.getenv("TELEGRAM_POOL_SIZE", "8")),
}
