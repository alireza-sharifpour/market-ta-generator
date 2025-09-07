import logging
from typing import Dict, Any

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
            response = AnalysisResponse(
                status=result["status"],
                analysis=result["analysis"],
                analysis_summarized=result.get("analysis_summarized"),
                message=None,
                chart_image_base64=result.get("chart_image_base64"),
            )
            # Add debug logging for the final response
            logger.debug("======== FINAL RESPONSE SENT FROM ENDPOINT ========")
            logger.debug(f"Status: {response.status}")
            logger.debug(f"Analysis: {response.analysis}")
            logger.debug(f"Analysis Summarized: {response.analysis_summarized}")
            logger.debug(f"Message: {response.message}")
            logger.debug(f"Chart Image Base64: {'Present' if response.chart_image_base64 else 'None'}")
            logger.debug("==================================================")
            return response
        else:
            # Return error response with appropriate message
            response = AnalysisResponse(
                status=result["status"],
                analysis=None,
                analysis_summarized=None,
                message=result["message"],
                chart_image_base64=None,
            )
            # Add debug logging for the error response
            logger.debug("======== ERROR RESPONSE SENT FROM ENDPOINT ========")
            logger.debug(f"Status: {response.status}")
            logger.debug(f"Message: {response.message}")
            logger.debug("==================================================")
            return response
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during analysis: {str(e)}"
        )


@router.get("/cache-stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Endpoint for retrieving cache statistics and performance metrics.
    
    Returns:
        Dictionary containing cache statistics including hit rates, memory usage, etc.
    """
    try:
        from app.core.llm_cache import llm_cache
        stats = await llm_cache.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": stats,
            "message": "Cache statistics retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {str(e)}")
        return {
            "status": "error",
            "cache_stats": {"enabled": False, "connected": False},
            "message": f"Failed to retrieve cache stats: {str(e)}"
        }
