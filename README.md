# Market TA Generator 📈

A comprehensive cryptocurrency technical analysis service that generates AI-powered trading insights with visual charts and detailed reports.

## 🎯 Overview

Market TA Generator is a sophisticated service that analyzes cryptocurrency trading pairs by:

- Fetching real-time market data from LBank exchange
- Calculating technical indicators (EMAs, RSI, Bollinger Bands, ADX, MFI)
- Identifying support and resistance levels using clustering algorithms
- Generating AI-powered analysis using Avalai/Gemini models
- Creating visual charts with technical indicators and S/R levels
- Providing both detailed English analysis and summarized Persian reports

## ⚡ Key Features

- **🔍 Multi-Phase Analysis Pipeline**: 7-step comprehensive analysis process
- **📊 Technical Indicators**: EMAs (9, 21, 50), RSI, Bollinger Bands, ADX, MFI
- **🎯 Support/Resistance Detection**: Automated clustering-based level identification
- **🤖 AI-Powered Analysis**: Avalai/Gemini integration for intelligent market insights
- **📈 Chart Generation**: OHLCV charts with indicators and price levels
- **📊 Volume Analysis**: Docker-based automated batch processing of top 200 pairs with suspicious volume detection
- **🌐 Dual Language Support**: English detailed analysis + Persian summaries
- **🔒 Security**: IP whitelisting and rate limiting
- **📊 Monitoring**: Prometheus, Grafana, Loki integration
- **⚡ Caching**: Redis-based caching for performance
- **🐳 Containerized**: Full Docker deployment with monitoring stack

## 🏗️ Architecture

```
┌─────────────────┐      HTTP POST       ┌──────────────────────────┐      API Call       ┌─────────────┐
│ External Service│  (e.g., /analyze)   │    Market TA Generator   │───────────────────→│ LBank API   │
│ (e.g., Telegram │────────────────────→│      (FastAPI App)       │←───────────────────│ (OHLCV Data)│
│ Bot)            │                     │                          │                     └─────────────┘
└─────────────────┘      JSON Response   │  ┌─────────────────────┐ │
                     ←──────────────────  │  │ API Handler         │ │      API Call       ┌─────────────┐
                                         │  └─────────────────────┘ │───────────────────→│ Avalai API  │
                                         │  ┌─────────────────────┐ │←───────────────────│ (Analysis)  │
                                         │  │ Data Processor      │ │                     └─────────────┘
                                         │  │ (pandas, pandas-ta) │ │
                                         │  └─────────────────────┘ │      Cache          ┌─────────────┐
                                         │  ┌─────────────────────┐ │←──────────────────→│ Redis       │
                                         │  │ Chart Generator     │ │                     └─────────────┘
                                         │  │ (mplfinance)        │ │
                                         │  └─────────────────────┘ │
                                         └──────────────────────────┘
```

### Core Components

- **API Layer** (`app/api/`): FastAPI application with analysis endpoint
- **Analysis Service** (`app/core/analysis_service.py`): Main orchestrator for the analysis pipeline
- **Data Processor** (`app/core/data_processor.py`): Technical indicator calculations using pandas-ta
- **Chart Generator** (`app/core/chart_generator.py`): Visual chart creation with mplfinance
- **LBank Client** (`app/external/lbank_client.py`): Exchange API integration
- **LLM Client** (`app/external/llm_client.py`): AI model integration (Avalai/Gemini)
- **Cache Service** (`app/core/cache_service.py`): Redis-based caching layer
- **Middleware**: IP whitelist, rate limiting, logging

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- API Keys:
  - Avalai API key (required)
  - LBank API key & secret (optional, for authenticated requests)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd market-ta-generator

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker (Recommended)

```bash
# Start full stack with monitoring
docker-compose up --build

# Or run just the app
docker-compose up app redis
```

### 3. Run Locally for Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/

# Analyze a trading pair
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{"pair": "eth_usdt"}'
```

## 📋 API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### Analyze Trading Pair

```http
POST /api/v1/analyze
```

**Request Body:**

```json
{
  "pair": "eth_usdt", // Required: Trading pair (e.g., "eth_usdt", "btc_usdt")
  "timeframe": "day1", // Optional: Candle timeframe (default: "day1")
  "limit": 200 // Optional: Number of candles (1-2000, default: 200)
}
```

**Timeframe Options:**

- `minute1`, `minute5`, `minute15`, `minute30`
- `hour1`, `hour4`, `hour8`, `hour12`
- `day1`, `week1`, `month1`

**Response:**

```json
{
  "status": "success",
  "analysis": "Detailed technical analysis in English...",
  "analysis_summarized": "خلاصه تحلیل به زبان فارسی...",
  "message": null,
  "chart_image_base64": "data:image/png;base64,iVBORw0KG..."
}
```


#### Volume Analysis Scheduler

```http
POST /api/v1/scheduler/start
```

Start the volume analysis scheduler.

**Query Parameters:**
- `schedule_type`: "interval" or "cron" (default: "interval")

**Response:**
```json
{
  "status": "success",
  "message": "Scheduler started with interval schedule",
  "schedule_type": "interval"
}
```

```http
POST /api/v1/scheduler/stop
```

Stop the volume analysis scheduler.

**Response:**
```json
{
  "status": "success",
  "message": "Scheduler stopped successfully"
}
```

```http
GET /api/v1/scheduler/status
```

Get current scheduler status and job information.

**Response:**
```json
{
  "status": "running",
  "is_running": true,
  "jobs": [
    {
      "id": "volume_analysis_job",
      "name": "Volume Analysis Batch Job",
      "next_run_time": "2025-01-14T10:35:00",
      "trigger": "interval[0:05:00]"
    }
  ],
  "scheduler_config": {
    "interval_minutes": 5,
    "max_instances": 1,
    "coalesce": true,
    "misfire_grace_time": 60
  }
}
```

```http
POST /api/v1/scheduler/run-manual
```

Run volume analysis manually (outside of schedule).

**Response:**
```json
{
  "success": true,
  "timestamp": "2025-01-14T10:30:00",
  "type": "manual"
}
```

#### Cache Statistics

```http
GET /api/v1/cache-stats
```

Returns cache performance metrics and statistics.

### Error Responses

| Status | Description         | Cause                               |
| ------ | ------------------- | ----------------------------------- |
| 400    | Bad Request         | Invalid payload, limit out of range |
| 403    | Forbidden           | IP not whitelisted                  |
| 404    | Not Found           | Trading pair not found              |
| 500    | Server Error        | Processing error, API failures      |
| 503    | Service Unavailable | External APIs unavailable           |

## 🛠️ Development Guide

### Project Structure

```
market-ta-generator/
├── app/
│   ├── api/              # FastAPI routes and models
│   ├── core/             # Business logic and services
│   ├── external/         # External API clients
│   ├── middleware/       # Security and rate limiting
│   ├── utils/            # Helpers and logging
│   └── main.py           # FastAPI application
├── monitoring/           # Grafana, Prometheus, Loki configs
├── tests/               # Test files
├── docs/                # Additional documentation
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Full stack deployment
├── CLAUDE.md          # Development instructions for Claude
└── README.md          # This file
```

### Development Setup

1. **Virtual Environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Tests:**

   ```bash
   # All tests
   python -m pytest

   # Specific test files
   python test_lbank_client.py
   python test_data_processor.py
   python test_analysis.py
   ```

4. **Development Server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Code Style

- Follow PEP 8 conventions
- Use type hints
- Comprehensive error handling
- Structured logging
- Modular design patterns

## ⚙️ Configuration

### Environment Variables

```bash
# Required
AVALAI_API_KEY="your_avalai_key"         # AI analysis service

# Optional
LBANK_API_KEY="your_lbank_key"           # LBank authentication
LBANK_API_SECRET="your_lbank_secret"     # LBank authentication
WHITELIST_ENABLED="True"                 # Enable IP filtering
WHITELISTED_IPS="127.0.0.1,10.0.0.0/8" # Allowed IP addresses

# Monitoring (for Grafana integration)
GRAFANA_URL="http://localhost:3000"      # Grafana dashboard URL
GRAFANA_API_KEY="your_grafana_token"     # Grafana service account token
```

### Technical Indicators Configuration

The system calculates these indicators (configured in `app/config.py`):

- **EMAs**: 9, 21, 50 periods (only EMA9, EMA50 shown on charts)
- **RSI**: 14 periods
- **Bollinger Bands**: 20 periods, 2 standard deviations
- **ADX**: 14 periods
- **MFI**: 14 periods

## 📊 Volume Analysis

### Overview

The integrated volume analysis feature automatically processes the top 200 trading pairs every 5 minutes to detect suspicious volume activity using statistical methods. This functionality was integrated from a dedicated volume analysis project to provide comprehensive market analysis alongside traditional technical indicators.

### Detection Method

**Mean + 4×Standard Deviation Method:**
- Calculates rolling mean and standard deviation of volume over a 25-period window
- Detects volume spikes when: `Volume > Mean + 4×Standard_Deviation`
- Provides statistical significance for volume anomaly detection
- Generates confidence scores based on spike severity

### Features

- **Automated Batch Processing**: Analyzes top 200 pairs every 5 minutes
- **Statistical Volume Detection**: Mathematically rigorous spike detection
- **Interactive Charts**: Plotly-based charts with suspicious period highlighting
- **Confidence Scoring**: Algorithmic confidence assessment (0.0 to 1.0)
- **Multiple Output Formats**: HTML charts, PNG images, and JSON data
- **Alert Generation**: Automated alerts for different spike severities
- **Cron/Systemd Integration**: Reliable scheduled execution

### Setup

#### Option 1: Cron Job (Simple & Reliable)

```bash
cd /path/to/market-ta-generator
./scripts/setup_volume_analysis_cron.sh
```

#### Option 2: Systemd Service (System Integration)

```bash
cd /path/to/market-ta-generator
sudo ./scripts/setup_volume_analysis_systemd.sh
```

#### Option 3: Python APScheduler (Advanced & Integrated)

```bash
# Standalone scheduler service
cd /path/to/market-ta-generator
source venv/bin/activate
python volume_scheduler_service.py --schedule-type interval --interval-minutes 5

# Or as part of FastAPI application
uvicorn app.main:app --reload
# Then use API endpoints to control scheduler
```

#### Manual Execution

```bash
cd /path/to/market-ta-generator
source venv/bin/activate
python volume_analysis_cron.py
```

### Monitoring

#### Cron/Systemd Monitoring
```bash
# View cron logs
tail -f logs/volume_analysis_cron.log

# View systemd logs
sudo journalctl -u volume-analysis.service -f

# Check cron job status
crontab -l | grep volume_analysis

# Check systemd timer status
sudo systemctl status volume-analysis.timer
```

#### APScheduler Monitoring
```bash
# Check scheduler status via API
curl http://localhost:8000/api/v1/scheduler/status

# Start scheduler via API
curl -X POST http://localhost:8000/api/v1/scheduler/start

# Stop scheduler via API
curl -X POST http://localhost:8000/api/v1/scheduler/stop

# Run manual analysis via API
curl -X POST http://localhost:8000/api/v1/scheduler/run-manual
```

### Output Files

The volume analysis generates files in `./volume_analysis_results/`:

- **Analysis Data** (`{pair}_{timeframe}_{timestamp}_data.json`): Raw analysis data and metrics
- **Interactive Chart** (`{pair}_{timeframe}_{timestamp}_chart.html`): Interactive Plotly chart
- **Full Report** (`{pair}_{timeframe}_{timestamp}_report.html`): Comprehensive HTML report
- **Batch Summary** (`batch_summary_{timestamp}.json`): Overall batch analysis summary

### Configuration

Volume analysis parameters can be customized in `app/config.py`:

```python
VOLUME_ANALYSIS_CONFIG = {
    "enable_mean_std_detection": True,  # Enable mean+std method
    "mean_std_lookback_period": 25,     # Rolling window size
    "mean_std_multiplier": 4.0,         # Standard deviation multiplier
    "confidence_threshold": 0.7,        # Minimum confidence for alerts
}

# Batch processing configuration
BATCH_CONFIG = {
    "pairs": TOP_200_PAIRS,             # Top 200 trading pairs
    "timeframe": "minute5",             # 5-minute timeframes
    "periods": 50,                      # Number of periods to analyze
    "max_concurrent": 10,               # Concurrent analysis limit
    "output_dir": "./volume_analysis_results",
}
```

## 🧪 Testing

### Test Structure

```bash
# Unit Tests
test_data_processor.py      # Data processing logic
test_lbank_client.py        # Exchange API client
tests/test_prepare_llm_input.py  # LLM input preparation

# Integration Tests
test_analysis.py            # Full analysis pipeline
test_integration.py         # End-to-end API testing
test_sr_improvement.py      # Support/resistance detection

# Data Tests
test_data_only.py           # Data fetching and validation

# Volume Analysis Tests
test_volume_analysis.py     # Complete volume analysis test with outputs
simple_volume_test.py       # Component import and initialization tests
```

### Running Tests

```bash
# All tests
python -m pytest

# Specific categories
python -m pytest tests/ -v              # Unit tests
python test_analysis.py                 # Integration test
python test_lbank_client.py            # API client test

# Volume analysis tests
python test_volume_analysis.py         # Full volume analysis test
python simple_volume_test.py           # Component tests

# With coverage
python -m pytest --cov=app tests/
```

## 🚀 Deployment

### Docker Deployment

The project includes a complete monitoring stack:

```bash
# Full stack (app + monitoring)
docker-compose up -d

# Just the application services
docker-compose up app redis -d

# View logs
docker-compose logs -f app
```

### Services Included

| Service           | Port | Purpose                  |
| ----------------- | ---- | ------------------------ |
| **app**           | 8000 | Main FastAPI application |
| **redis**         | 6379 | Caching layer            |
| **prometheus**    | 9090 | Metrics collection       |
| **grafana**       | 3000 | Monitoring dashboards    |
| **loki**          | 3100 | Log aggregation          |
| **node-exporter** | 9100 | System metrics           |
| **cadvisor**      | 8080 | Container metrics        |

### Production Considerations

- **Environment Variables**: Use secrets management
- **IP Whitelisting**: Configure allowed IPs
- **Rate Limiting**: Adjust based on usage patterns
- **Resource Limits**: Set appropriate Docker limits
- **Log Rotation**: Configure log retention policies
- **Backup**: Regular backup of Redis data

## 📊 Monitoring

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin:MarketTA2025!Secure#Grafana)

- **FastAPI Metrics**: Request rates, response times, error rates
- **System Metrics**: CPU, memory, disk usage
- **Container Metrics**: Docker container performance
- **Cache Metrics**: Redis performance and hit rates

### Prometheus Metrics

- Application metrics at `http://localhost:8000/metrics`
- System metrics via node-exporter
- Container metrics via cAdvisor

### Log Aggregation

Loki collects logs from:

- Application logs (structured JSON)
- System logs
- Container logs
- Docker daemon logs

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**

   ```
   Error: Invalid Avalai API key
   Solution: Check AVALAI_API_KEY in .env file
   ```

2. **Rate Limiting**

   ```
   Error: Too Many Requests (429)
   Solution: Implement exponential backoff, check rate limits
   ```

3. **IP Whitelist Blocked**

   ```
   Error: 403 Forbidden
   Solution: Add IP to WHITELISTED_IPS or disable WHITELIST_ENABLED
   ```

4. **Redis Connection**

   ```
   Error: Redis connection failed
   Solution: Ensure Redis is running, check connection settings
   ```

5. **Chart Generation**
   ```
   Error: Chart generation failed
   Solution: Check matplotlib dependencies, font availability
   ```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uvicorn app.main:app --reload

# Check service health
curl http://localhost:8000/
curl http://localhost:8000/api/v1/cache-stats
```

### Performance Tuning

- **Cache TTL**: Adjust Redis cache expiration based on data freshness needs
- **Concurrent Limits**: Tune rate limiting based on infrastructure
- **Memory Allocation**: Adjust Docker memory limits for containers
- **API Timeouts**: Configure appropriate timeouts for external APIs

## 🤝 Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes following code style guidelines
4. **Add** tests for new functionality
5. **Run** tests and ensure they pass
6. **Commit** changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Code Review Process

- All PRs require review
- Tests must pass
- Code coverage should not decrease
- Follow existing code patterns
- Update documentation as needed

### Areas for Contribution

- **New Exchanges**: Add support for additional exchanges beyond LBank
- **Technical Indicators**: Implement additional indicators
- **AI Models**: Integrate alternative LLM providers
- **Performance**: Optimize analysis pipeline performance
- **Documentation**: Improve documentation and examples
- **Testing**: Expand test coverage and integration tests

## 📞 Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Documentation**: Check `/docs` folder for additional guides
- **API Reference**: See `api_documentation.md` for detailed API docs
- **Architecture**: Review `docs/technical_PRD.md` for technical specifications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using FastAPI, pandas-ta, mplfinance, and AI-powered analysis**
