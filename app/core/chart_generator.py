import base64
import io
import logging
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Set the matplotlib backend to Agg for headless environments
matplotlib.use("Agg")


def generate_ohlcv_chart(
    df: pd.DataFrame,
    indicators_to_plot: Optional[List[str]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
) -> str:
    """
    Generate an OHLCV chart with candlesticks and optional technical indicators.

    This function creates a professional-looking candlestick chart using mplfinance,
    which is specifically designed for financial data visualization. The chart includes:
    - Candlestick price data (OHLC)
    - Volume bars below the price chart
    - Optional technical indicators overlay

    Args:
        df: DataFrame with OHLCV data and datetime index
            Must have columns: Open, High, Low, Close, Volume
        indicators_to_plot: Optional list of indicator column names to overlay
                           (e.g., ['EMA_20', 'EMA_50', 'RSI_14'])
        chart_style: Style for the chart ('yahoo', 'charles', 'tradingview', etc.)
        figsize: Tuple specifying the figure size (width, height)

    Returns:
        Base64 encoded string representation of the chart image
        Format: "data:image/png;base64,<encoded_data>"

    Raises:
        ValueError: If the DataFrame is empty or missing required columns
        RuntimeError: If chart generation fails
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot generate chart.")
        raise ValueError("DataFrame cannot be empty")

    # Validate required columns
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"DataFrame missing required columns: {missing}")
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Validate datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be DatetimeIndex")
        raise ValueError("DataFrame index must be DatetimeIndex")

    try:
        logger.info(f"Generating OHLCV chart for {len(df)} data points")
        logger.info(f"Data range: {df.index.min()} to {df.index.max()}")

        # Prepare additional plots for indicators
        addplot_list = []

        if indicators_to_plot:
            logger.info(f"Adding indicators to chart: {indicators_to_plot}")

            for indicator in indicators_to_plot:
                if indicator in df.columns:
                    # Check if the indicator has valid data
                    if not df[indicator].isna().all():
                        # Create addplot for each indicator
                        addplot_list.append(
                            mpf.make_addplot(
                                df[indicator], type="line", width=1.5, alpha=0.8
                            )
                        )
                        logger.info(f"Added indicator {indicator} to chart")
                    else:
                        logger.warning(
                            f"Indicator {indicator} has no valid data, skipping"
                        )
                else:
                    logger.warning(
                        f"Indicator {indicator} not found in DataFrame, skipping"
                    )

        # Set up the chart configuration
        chart_config = {
            "type": "candle",
            "style": chart_style,
            "volume": True,  # Always include volume
            "figsize": figsize,
            "tight_layout": True,
            "returnfig": True,  # Return figure for base64 encoding
        }

        # Add indicators if any are available
        if addplot_list:
            chart_config["addplot"] = addplot_list

        # Generate the chart using mplfinance
        logger.info("Generating chart with mplfinance...")
        fig, axes = mpf.plot(df, **chart_config)

        # Convert the figure to base64 encoded string
        logger.info("Converting chart to base64 format...")
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Clean up
        plt.close(fig)
        buffer.close()

        # Format as data URL
        chart_data_url = f"data:image/png;base64,{image_base64}"

        logger.info("Chart generated successfully")
        logger.info(f"Chart data URL length: {len(chart_data_url)} characters")

        return chart_data_url

    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        raise RuntimeError(f"Failed to generate chart: {str(e)}")


def generate_ohlcv_chart_with_bollinger_bands(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
) -> str:
    """
    Generate an OHLCV chart with Bollinger Bands overlay.

    This is a convenience function specifically for displaying Bollinger Bands,
    which are commonly used in technical analysis.

    Args:
        df: DataFrame with OHLCV data and datetime index
        bb_period: Period for Bollinger Bands calculation
        bb_std: Standard deviation multiplier for Bollinger Bands
        chart_style: Style for the chart
        figsize: Figure size

    Returns:
        Base64 encoded string representation of the chart image
    """
    # Look for Bollinger Band columns in the DataFrame
    bb_upper_col = f"BBU_{bb_period}_{bb_std}"
    bb_middle_col = f"BBM_{bb_period}_{bb_std}"
    bb_lower_col = f"BBL_{bb_period}_{bb_std}"

    indicators: List[str] = []
    for col in [bb_upper_col, bb_middle_col, bb_lower_col]:
        if col in df.columns:
            indicators.append(col)

    if not indicators:
        logger.warning("No Bollinger Band columns found, generating chart without them")

    return generate_ohlcv_chart(
        df=df, indicators_to_plot=indicators, chart_style=chart_style, figsize=figsize
    )


def generate_ohlcv_chart_with_emas(
    df: pd.DataFrame,
    ema_periods: Optional[List[int]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
) -> str:
    """
    Generate an OHLCV chart with EMA overlays.

    Args:
        df: DataFrame with OHLCV data and datetime index
        ema_periods: List of EMA periods to include (e.g., [20, 50, 200])
        chart_style: Style for the chart
        figsize: Figure size

    Returns:
        Base64 encoded string representation of the chart image
    """
    if ema_periods is None:
        ema_periods = [20, 50, 200]  # Default EMAs

    # Look for EMA columns in the DataFrame
    ema_indicators = []
    for period in ema_periods:
        ema_col = f"EMA_{period}"
        if ema_col in df.columns:
            ema_indicators.append(ema_col)

    if not ema_indicators:
        logger.warning("No EMA columns found, generating chart without them")

    return generate_ohlcv_chart(
        df=df,
        indicators_to_plot=ema_indicators,
        chart_style=chart_style,
        figsize=figsize,
    )


def list_available_indicators(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    List all available technical indicators in the DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary categorizing available indicators by type
    """
    indicators: Dict[str, List[str]] = {
        "ema": [],
        "sma": [],
        "rsi": [],
        "bollinger": [],
        "adx": [],
        "mfi": [],
        "other": [],
    }

    for col in df.columns:
        if col in ["Open", "High", "Low", "Close", "Volume"]:
            continue  # Skip OHLCV columns

        col_lower = col.lower()
        if "ema_" in col_lower:
            indicators["ema"].append(col)
        elif "sma_" in col_lower:
            indicators["sma"].append(col)
        elif "rsi" in col_lower:
            indicators["rsi"].append(col)
        elif any(bb in col_lower for bb in ["bb", "bollinger"]):
            indicators["bollinger"].append(col)
        elif "adx" in col_lower or "dm" in col_lower:
            indicators["adx"].append(col)
        elif "mfi" in col_lower:
            indicators["mfi"].append(col)
        else:
            indicators["other"].append(col)

    # Remove empty categories
    return {k: v for k, v in indicators.items() if v}
