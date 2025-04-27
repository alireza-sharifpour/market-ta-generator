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
DEFAULT_SIZE = 60

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gemini-2.0-flash"

# IP Whitelist Configuration
# Default values provided, but can be overridden with environment variables
WHITELIST_ENABLED = os.getenv("WHITELIST_ENABLED", "True").lower() == "true"
# Define whitelisted IPs as a comma-separated string in the environment
WHITELISTED_IPS = os.getenv(
    "WHITELISTED_IPS", "127.0.0.1,154.90.55.18,10.72.24.67,4.210.246.79"
).split(",")
