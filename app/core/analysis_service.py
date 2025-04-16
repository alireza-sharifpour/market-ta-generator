"""
Analysis Service module that orchestrates the workflow between data fetching,
processing, and LLM analysis.
"""

import logging
from typing import Any, Dict

from app.core.data_processor import format_data_for_llm, process_raw_data
from app.external.lbank_client import LBankAPIError, LBankConnectionError, fetch_ohlcv
from app.external.llm_client import generate_basic_analysis

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Exception raised for errors in the analysis service."""

    pass


def run_phase1_analysis(pair: str) -> Dict[str, Any]:
    """
    Orchestrates the Phase 1 analysis workflow:
    1. Fetch raw OHLCV data from LBank API
    2. Process the raw data into a DataFrame
    3. Format the data for LLM consumption
    4. Generate a basic analysis using the LLM

    Args:
        pair: Trading pair symbol (e.g., "eth_usdt")

    Returns:
        Dictionary with analysis results containing:
        - "status": "success" or "error"
        - "analysis": The generated analysis text (if successful)
        - "message": Error message (if failed)

    Raises:
        AnalysisError: If any part of the analysis process fails
    """
    try:
        logger.info(f"Starting Phase 1 analysis for pair: {pair}")

        # Step 1: Fetch raw OHLCV data from LBank API
        logger.info(f"Fetching OHLCV data for {pair}")
        try:
            raw_data = fetch_ohlcv(pair)
            logger.info(f"Successfully fetched {len(raw_data)} data points for {pair}")
        except (LBankAPIError, LBankConnectionError) as e:
            error_msg = f"Failed to fetch data from LBank: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Step 2: Process the raw data into a DataFrame
        logger.info("Processing raw data into DataFrame")
        try:
            df = process_raw_data(raw_data)
            logger.info(
                f"Successfully processed data into DataFrame with {len(df)} rows"
            )
        except ValueError as e:
            error_msg = f"Failed to process raw data: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Step 3: Format the data for LLM consumption
        logger.info("Formatting data for LLM")
        try:
            formatted_data = format_data_for_llm(df)
            logger.info("Successfully formatted data for LLM")
        except Exception as e:
            error_msg = f"Failed to format data for LLM: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Step 4: Generate analysis using LLM
        logger.info("Generating analysis using LLM")
        try:
            analysis_text = generate_basic_analysis(pair, formatted_data)
            logger.info("Successfully generated analysis")
        except Exception as e:
            error_msg = f"Failed to generate analysis with LLM: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Return successful result
        return {"status": "success", "analysis": analysis_text, "message": None}

    except Exception as e:
        error_msg = f"Unexpected error during analysis: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
