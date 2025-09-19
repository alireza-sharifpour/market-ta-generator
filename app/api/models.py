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
    analysis_summarized: Optional[str] = Field(
        None, description="Short summarized analysis in Persian format"
    )
    message: Optional[str] = Field(None, description="Error message in case of failure")
    chart_image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded chart image (Phase 3). Format: data:image/png;base64,...",
    )


class VolumeAnalysisRequest(BaseModel):
    """
    Model representing the request payload for the /volume-analyze endpoint.
    """

    pair: str = Field(..., description="The trading pair symbol (e.g., 'ETHUSDT')")
    timeframe: Optional[str] = Field(
        None,
        description="The timeframe for OHLCV data (e.g., 'minute1', 'hour4', 'day1')",
    )
    limit: Optional[int] = Field(
        None, description="Number of candles to fetch (1-2000)", ge=1, le=2000
    )


class VolumeAnalysisResponse(BaseModel):
    """
    Model representing the response from the /volume-analyze endpoint.
    """

    status: str = Field(..., description="Response status (success or error)")
    pair: str = Field(..., description="Trading pair analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    analysis_timestamp: str = Field(..., description="When the analysis was performed")
    
    # Analysis results
    suspicious_periods: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of suspicious periods detected"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis metrics and statistics"
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Generated alerts"
    )
    confidence_score: float = Field(
        0.0, description="Confidence score of the analysis (0.0 to 1.0)"
    )
    
    # Chart and report outputs
    chart_html: Optional[str] = Field(
        None, description="HTML content of the interactive chart"
    )
    chart_image_base64: Optional[str] = Field(
        None, description="Base64 encoded chart image"
    )
    report_html: Optional[str] = Field(
        None, description="Full HTML analysis report"
    )
    
    # Error handling
    message: Optional[str] = Field(None, description="Error message in case of failure")
