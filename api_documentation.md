# Market TA Generator API Documentation

## Overview

This document provides guidelines for front-end developers to integrate with the Market TA Generator API. The API generates comprehensive technical analysis for cryptocurrency trading pairs by fetching data from the LBank exchange, performing technical analysis with multiple indicators, and using AI (Avalai/Gemini) to analyze market conditions. The service includes chart generation with visual support/resistance levels and provides both detailed and summarized analysis outputs.

## Base URL

```
http://localhost:8000/api/v1
```

When deployed, replace with the appropriate host and port.

## Authentication & Security

The API doesn't require authentication from the client. The service itself handles authentication with LBank and Avalai using API keys configured in the backend.

**IP Whitelist**: The API includes IP-based access control that can be configured via environment variables. Requests from non-whitelisted IPs will receive a 403 Forbidden response.

## Endpoints

### Health Check

```
GET /
```

Used to verify the service is running properly.

**Response:**

```json
{
  "status": "ok",
  "service": "Market TA Generator"
}
```

### Generate Technical Analysis

```
POST /api/v1/analyze
```

Analyzes a cryptocurrency trading pair using historical market data.

**Request Body:**

```json
{
  "pair": "string",     // Required: Trading pair (e.g., "eth_usdt")
  "timeframe": "string", // Optional: Candle timeframe
  "limit": integer      // Optional: Number of candles to fetch (1-2000)
}
```

**Parameters:**

| Parameter | Type    | Required | Description                                        | Default Value |
| --------- | ------- | -------- | -------------------------------------------------- | ------------- |
| pair      | string  | Yes      | Trading pair symbol (e.g., "eth_usdt", "btc_usdt") | N/A           |
| timeframe | string  | No       | Candle timeframe (see valid values below)          | "day1"        |
| limit     | integer | No       | Number of candles to fetch (range: 1-2000)         | 200           |

**Valid Timeframe Values:**

- "minute1" - 1 minute
- "minute5" - 5 minutes
- "minute15" - 15 minutes
- "minute30" - 30 minutes
- "hour1" - 1 hour
- "hour4" - 4 hours
- "hour8" - 8 hours
- "hour12" - 12 hours
- "day1" - 1 day
- "week1" - 1 week
- "month1" - 1 month

**Successful Response (200 OK):**

```json
{
  "status": "success",
  "analysis": "Generated detailed technical analysis text...",
  "analysis_summarized": "خلاصه تحلیل به زبان فارسی...",
  "message": null,
  "chart_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUh..."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| status | string | Always "success" for successful requests |
| analysis | string | Detailed technical analysis in English |
| analysis_summarized | string | Short summarized analysis in Persian |
| message | null | Always null for successful responses |
| chart_image_base64 | string | Base64 encoded PNG chart image with technical indicators and support/resistance levels |

**Error Response (400, 403, 404, 500):**

```json
{
  "status": "error",
  "analysis": null,
  "analysis_summarized": null,
  "message": "Error description",
  "chart_image_base64": null
}
```

## Error Codes

| HTTP Status | Description           | Possible Causes                                     |
| ----------- | --------------------- | --------------------------------------------------- |
| 400         | Bad Request           | Invalid input payload, malformed request, limit out of range (1-2000) |
| 403         | Forbidden             | IP address not whitelisted (when IP whitelist is enabled) |
| 404         | Not Found             | Trading pair not found on LBank                     |
| 500         | Internal Server Error | Server error, Avalai API error, processing error    |
| 503         | Service Unavailable   | LBank API or Avalai API unavailable or rate limited |

## Example Usage

### JavaScript Fetch Example

```javascript
// Function to analyze a cryptocurrency pair
async function analyzeCrypto(pair, timeframe = null, limit = null) {
  const requestBody = {
    pair: pair,
  };

  // Add optional parameters if provided
  if (timeframe) requestBody.timeframe = timeframe;
  if (limit) requestBody.limit = limit;

  try {
    const response = await fetch("http://localhost:8000/api/v1/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await response.json();

    if (response.ok) {
      // Handle successful response
      console.log("Detailed Analysis:", data.analysis);
      console.log("Summary (Persian):", data.analysis_summarized);
      
      // Display chart if needed
      if (data.chart_image_base64) {
        const img = document.createElement("img");
        img.src = data.chart_image_base64;
        document.body.appendChild(img);
      }
      
      return data;
    } else {
      // Handle error response
      console.error("Error:", data.message);
      throw new Error(data.message);
    }
  } catch (error) {
    console.error("API request failed:", error);
    throw error;
  }
}
```

## Analysis Pipeline

The API follows a comprehensive 7-step analysis process:

1. **Data Fetching** - Retrieves OHLCV data from LBank API
2. **Data Processing** - Converts raw data to DataFrame
3. **Technical Indicators** - Calculates EMAs (9, 21, 50), RSI, Bollinger Bands, ADX, MFI
4. **Support/Resistance Analysis** - Identifies key price levels using clustering algorithms
5. **Current Price Fetch** - Gets live market price (supplementary)
6. **AI Analysis** - Generates detailed analysis using Avalai/Gemini model
7. **Chart Generation** - Creates OHLCV chart with EMA9, EMA50, and S/R levels

## Notes for Front-End Development

1. **Trading Pair Format**: Use lowercase format with underscore separator (e.g., "eth_usdt", "btc_usdt"). This matches LBank's expected format.

2. **Response Time**: Expect responses to take 10-20 seconds as the process involves data fetching, technical analysis calculation, AI processing, and chart generation.

3. **Chart Integration**: The returned `chart_image_base64` field contains a complete data URL that can be directly used as an image source in HTML (`<img src="data:image/png;base64,..."`).

4. **Dual Language Support**: The API provides both detailed English analysis and summarized Persian analysis for different user audiences.

5. **Error Handling**: Implement robust error handling for all API calls. Pay special attention to 403 responses (IP whitelist) and 400 responses (validation errors).

6. **Rate Limiting**: Be aware the backend may be subject to rate limiting from both LBank and Avalai APIs. Implement retry logic with exponential backoff if needed.

7. **IP Whitelist**: If deploying, ensure your front-end server's IP is added to the whitelist or disable IP filtering via environment variables.

8. **Technical Indicators**: While the chart displays only EMA9 and EMA50 for clarity, the analysis considers all calculated indicators (RSI, Bollinger Bands, ADX, MFI).

9. **Support/Resistance Levels**: The generated charts include visual markers for identified support and resistance levels, which are also referenced in the analysis text.

## Environment Variables (Backend Configuration)

```bash
LBANK_API_KEY="your_lbank_key"           # Optional - for authenticated requests
LBANK_API_SECRET="your_lbank_secret"     # Optional - for authenticated requests  
AVALAI_API_KEY="your_avalai_key"         # Required - for AI analysis (Gemini 2.5 Flash)
WHITELIST_ENABLED="True"                 # Optional - defaults to True
WHITELISTED_IPS="127.0.0.1,..."        # Optional - comma-separated IPs
```
