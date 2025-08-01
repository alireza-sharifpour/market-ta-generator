import logging

from fastapi import APIRouter, HTTPException

from app.api.models import AnalysisRequest, AnalysisResponse
from app.core.analysis_service import run_phase2_analysis

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_pair(request: AnalysisRequest):
    """
    Endpoint for generating technical analysis for a cryptocurrency pair.

    Args:
        request: The analysis request containing the trading pair.

    Returns:
        AnalysisResponse: The analysis result or error message.

    Raises:
        HTTPException: If any error occurs during the analysis process.
    """
    try:
        logger.info(f"Received analysis request for pair: {request.pair}")
        if request.timeframe:
            logger.info(f"Using timeframe: {request.timeframe}")
        if request.limit:
            logger.info(f"Using limit: {request.limit}")

        # Call the analysis service to process the request
        result = await run_phase2_analysis(
            request.pair, timeframe=request.timeframe, limit=request.limit
        )

        # Check if the analysis was successful
        if result["status"] == "success":
            return AnalysisResponse(
                status=result["status"],
                analysis=result["analysis"],
                analysis_summarized=result.get("analysis_summarized"),
                message=None,
                chart_image_base64=result.get("chart_image_base64"),
            )
        else:
            # Return error response with appropriate message
            return AnalysisResponse(
                status=result["status"],
                analysis=None,
                analysis_summarized=None,
                message=result["message"],
                chart_image_base64=None,
            )
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during analysis: {str(e)}"
        )
