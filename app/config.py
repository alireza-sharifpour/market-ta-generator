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
OPENAI_MODEL = "gpt-4o"
