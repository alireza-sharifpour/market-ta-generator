# Market TA Generator API Documentation

## Overview

This document provides guidelines for front-end developers to integrate with the Market TA Generator API. The API allows you to generate technical analysis for cryptocurrency trading pairs by fetching data from the LBank exchange and using AI to analyze market conditions.

## Base URL

```
http://localhost:8000/api/v1
```

When deployed, replace with the appropriate host and port.

## Authentication

The API doesn't require authentication from the client. The service itself handles authentication with LBank and OpenAI using API keys configured in the backend.

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
| limit     | integer | No       | Number of candles to fetch (range: 1-2000)         | 60            |

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
  "analysis": "Generated technical analysis text...",
  "message": null
}
```

**Error Response (400, 404, 500):**

```json
{
  "status": "error",
  "analysis": null,
  "message": "Error description"
}
```

## Error Codes

| HTTP Status | Description           | Possible Causes                                     |
| ----------- | --------------------- | --------------------------------------------------- |
| 400         | Bad Request           | Invalid input payload, malformed request            |
| 404         | Not Found             | Trading pair not found on LBank                     |
| 500         | Internal Server Error | Server error, OpenAI API error, processing error    |
| 503         | Service Unavailable   | LBank API or OpenAI API unavailable or rate limited |

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
      console.log("Analysis:", data.analysis);
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

## Notes for Front-End Development

1. **Trading Pair Format**: LBank typically uses lowercase symbols with an underscore separator (e.g., "eth_usdt" not "ETHUSDT"). Check LBank documentation for specific pair formats.

2. **Response Time**: Expect responses to take several seconds (up to 15 seconds) as the process involves fetching data from LBank and generating analysis with the AI model.

3. **Error Handling**: Implement robust error handling for all API calls, displaying appropriate messages to users.

4. **Rate Limiting**: Be aware the backend may be subject to rate limiting from both LBank and OpenAI. Implement retry logic if needed.

5. **Formatting Analysis Output**: The generated analysis is returned as text. Consider implementing formatting (e.g. parsing) to improve readability for users.
