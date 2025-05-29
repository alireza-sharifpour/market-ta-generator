"""
Analysis Service module that orchestrates the workflow between data fetching,
processing, and LLM analysis.
"""

import logging
from typing import Any, Dict, Optional

from app.config import DEFAULT_SIZE, DEFAULT_TIMEFRAME
from app.core.chart_generator import generate_ohlcv_chart
from app.core.data_processor import (
    add_technical_indicators,
    identify_support_resistance,
    prepare_llm_input_phase2,
    process_raw_data,
)
from app.external.lbank_client import LBankAPIError, LBankConnectionError, fetch_ohlcv
from app.external.llm_client import (
    generate_detailed_analysis,
)

# Set up logging
logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Exception raised for errors in the analysis service."""

    pass


def run_phase2_analysis(
    pair: str, timeframe: Optional[str] = None, limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Orchestrates the Phase 2 analysis workflow:
    1. Fetch raw OHLCV data from LBank API
    2. Process the raw data into a DataFrame
    3. Add technical indicators to the DataFrame
    4. Identify Support/Resistance levels
    5. Prepare structured data for LLM consumption
    6. Generate a detailed analysis using the LLM

    Args:
        pair: Trading pair symbol (e.g., "eth_usdt")
        timeframe: Time interval for each candle (e.g., "day1", "hour4")
                   Defaults to DEFAULT_TIMEFRAME if None
        limit: Number of candles to fetch (1-2000)
               Defaults to DEFAULT_SIZE if None

    Returns:
        Dictionary with analysis results containing:
        - "status": "success" or "error"
        - "analysis": The generated detailed analysis text (if successful)
        - "message": Error message (if failed)
    """
    try:
        timeframe_to_use = DEFAULT_TIMEFRAME if timeframe is None else timeframe
        limit_to_use = DEFAULT_SIZE if limit is None else limit

        logger.info(
            f"Starting Phase 2 analysis for pair: {pair}, timeframe: {timeframe_to_use}, limit: {limit_to_use}"
        )

        # Step 1: Fetch raw OHLCV data
        logger.info(f"Fetching OHLCV data for {pair}")
        try:
            raw_data = fetch_ohlcv(pair, timeframe=timeframe_to_use, limit=limit_to_use)
            logger.info(f"Successfully fetched {len(raw_data)} data points for {pair}")
        except (LBankAPIError, LBankConnectionError) as e:
            error_msg = f"Failed to fetch data from LBank: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:  # Catch any other unexpected error during fetch
            error_msg = f"Unexpected error fetching data from LBank: {str(e)}"
            logger.error(error_msg, exc_info=True)
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
        except Exception as e:  # Catch any other unexpected error during processing
            error_msg = f"Unexpected error processing raw data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        # Step 3: Add technical indicators
        logger.info("Adding technical indicators")
        try:
            df_with_indicators = add_technical_indicators(df)
            logger.info("Successfully added technical indicators")
        except (ValueError, RuntimeError) as e:
            error_msg = f"Failed to add technical indicators: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:  # Catch any other unexpected error
            error_msg = f"Unexpected error adding technical indicators: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        # Step 4: Identify Support/Resistance levels
        logger.info("Identifying S/R levels")
        try:
            # Using default parameters for identify_support_resistance
            sr_levels = identify_support_resistance(df_with_indicators)
            logger.info(
                f"Successfully identified S/R levels: Support {sr_levels.get('support')}, Resistance {sr_levels.get('resistance')}"
            )
        except (ValueError, RuntimeError) as e:
            error_msg = f"Failed to identify S/R levels: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:  # Catch any other unexpected error
            error_msg = f"Unexpected error identifying S/R levels: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        # Step 5: Prepare structured data for LLM
        logger.info("Preparing structured data for LLM (Phase 2)")
        try:
            structured_llm_input = prepare_llm_input_phase2(
                df_with_indicators, sr_levels
            )
            logger.info("Successfully prepared structured data for LLM")
        except ValueError as e:
            error_msg = f"Failed to prepare LLM input: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:  # Catch any other unexpected error
            error_msg = f"Unexpected error preparing LLM input: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        # Step 6: Generate detailed analysis using LLM
        logger.info("Generating detailed analysis using LLM (Phase 2)")
        try:
            analysis_text = generate_detailed_analysis(
                pair, structured_llm_input, timeframe=timeframe_to_use
            )
            logger.info("Successfully generated detailed analysis")
        except Exception as e:
            error_msg = f"Failed to generate detailed analysis with LLM: {str(e)}"
            logger.error(
                error_msg, exc_info=True
            )  # exc_info for more details on LLM errors
            return {"status": "error", "message": error_msg}

        # Generate chart with indicators (Phase 3)
        logger.info("Generating OHLCV chart with indicators")
        chart_image_base64 = None
        try:
            # Use the new filtered chart generation which will automatically
            # show only EMA50 and EMA9 while keeping all indicators for LLM analysis
            # Also pass the calculated S/R levels for visualization
            chart_image_base64 = generate_ohlcv_chart(
                df_with_indicators,
                indicators_to_plot=None,  # Let the chart generator use filtered indicators
                sr_levels=sr_levels,  # Pass the calculated S/R levels
                use_filtered_indicators=True,  # Use only EMA50 and EMA9
            )
            logger.info(
                "Successfully generated chart with filtered indicators (EMA50 and EMA9) and S/R levels"
            )
        except Exception as e:
            # Chart generation is not critical - log the error but continue
            error_msg = f"Failed to generate chart with indicators: {str(e)}"
            logger.warning(error_msg)
            chart_image_base64 = None

        return {
            "status": "success",
            "analysis": analysis_text,
            "message": None,
            "chart_image_base64": chart_image_base64,
        }

    except Exception as e:
        # This is a catch-all for unexpected errors not caught in specific steps
        error_msg = f"Unexpected error during Phase 2 analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}
