from typing import Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """
    Model representing the request payload for the /analyze endpoint.
    """

    pair: str = Field(..., description="The trading pair symbol (e.g., 'ETHUSDT')")
    timeframe: Optional[str] = Field(
        None,
        description="The timeframe for OHLCV data (e.g., 'minute1', 'hour4', 'day1')",
    )
    limit: Optional[int] = Field(
        None, description="Number of candles to fetch (1-2000)", ge=1, le=2000
    )


class AnalysisResponse(BaseModel):
    """
    Model representing the response from the /analyze endpoint.

    This model supports both Phase 1/2 responses (basic analysis) and Phase 3 responses
    (analysis with optional chart generation).
    """

    status: str = Field(..., description="Response status (success or error)")
    analysis: Optional[str] = Field(
        None, description="Generated technical analysis text"
    )
    message: Optional[str] = Field(None, description="Error message in case of failure")
    chart_image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded chart image (Phase 3). Format: data:image/png;base64,...",
    )
