from typing import Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """
    Model representing the request payload for the /analyze endpoint.
    """

    pair: str = Field(..., description="The trading pair symbol (e.g., 'ETHUSDT')")


class AnalysisResponse(BaseModel):
    """
    Model representing the response from the /analyze endpoint.
    """

    status: str = Field(..., description="Response status (success or error)")
    analysis: Optional[str] = Field(
        None, description="Generated technical analysis text"
    )
    message: Optional[str] = Field(None, description="Error message in case of failure")
