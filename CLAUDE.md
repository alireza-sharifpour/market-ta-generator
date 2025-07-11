# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run development server
source .venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with Docker
docker-compose up --build

# Health check
curl http://localhost:8000/
```

### Testing

```bash
# Run all tests with pytest
source .venv/bin/activate && python -m pytest

# Run specific test file
source .venv/bin/activate && python test_analysis.py

# Run individual test scripts
python test_data_processor.py
python test_lbank_client.py
python test_data_only.py
```

## Architecture Overview

### Core Workflow (Phase 2 Analysis)

The application follows a structured pipeline orchestrated by `analysis_service.py`:

1. **Data Fetching** (`lbank_client.py`) - Retrieves OHLCV data from LBank API
2. **Data Processing** (`data_processor.py`) - Converts raw data to DataFrame and adds technical indicators
3. **Support/Resistance Analysis** - Identifies key price levels using clustering algorithms
4. **LLM Analysis** (`llm_client.py`) - Generates technical analysis using Avalai API (OpenAI-compatible)
5. **Chart Generation** (`chart_generator.py`) - Creates OHLCV charts with indicators and S/R levels

### Key Components

#### API Layer (`app/api/`)

- **FastAPI application** with single `/api/v1/analyze` endpoint
- **IP whitelist middleware** for access control
- **Pydantic models** for request/response validation

#### Core Services (`app/core/`)

- **`analysis_service.py`** - Main orchestrator for the analysis pipeline
- **`data_processor.py`** - OHLCV data processing and technical indicator calculations using pandas-ta
- **`chart_generator.py`** - Chart generation using mplfinance with filtered indicators (EMA9, EMA50)

#### External Integrations (`app/external/`)

- **`lbank_client.py`** - LBank cryptocurrency exchange API client with authentication
- **`llm_client.py`** - Avalai/OpenAI compatible LLM client with retry logic and error handling

### Configuration (`app/config.py`)

- Environment variables loaded via python-dotenv
- API keys for LBank and Avalai services
- Technical indicator settings (EMA periods, RSI, Bollinger Bands, etc.)
- IP whitelist configuration

### Technical Indicators

The system calculates multiple indicators but **only displays EMA9 and EMA50 on charts** for clarity:

- EMAs (9, 21, 50 periods)
- RSI (14 period)
- Bollinger Bands (20 period, 2 std dev)
- ADX and MFI (14 period)

### Support/Resistance Detection

Uses clustering algorithms with optimized parameters:

- Reduced lookback periods for improved sensitivity
- Price level clustering to identify key support and resistance zones
- Integration with chart visualization

## Environment Variables Required

Copy `.env.example` to `.env` and configure:

```bash
LBANK_API_KEY="your_lbank_key"           # Optional - for authenticated requests
LBANK_API_SECRET="your_lbank_secret"     # Optional - for authenticated requests
AVALAI_API_KEY="your_avalai_key"         # Required - for LLM analysis
WHITELIST_ENABLED="True"                 # Optional - defaults to True
WHITELISTED_IPS="127.0.0.1,..."        # Optional - comma-separated IPs
```

## Error Handling

The application implements comprehensive error handling at each pipeline stage:

- **LBank API errors** - Network and API-specific exceptions
- **Data processing errors** - DataFrame validation and indicator calculation failures
- **LLM errors** - Rate limiting, authentication, and API connection issues
- **Chart generation errors** - Non-critical failures that don't stop the analysis

All errors are logged with appropriate levels and returned in structured API responses.

## Testing Strategy

- **Unit tests** for individual components (data processor, LBank client)
- **Integration tests** for the complete analysis pipeline
- **Manual test scripts** for API endpoint validation
- Tests are located both in root directory and `tests/` folder
