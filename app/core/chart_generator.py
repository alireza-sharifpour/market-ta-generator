import base64
import io
import logging
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore
import pandas as pd
from matplotlib.lines import Line2D

# Set up logging
logger = logging.getLogger(__name__)

# Set the matplotlib backend to Agg for headless environments
matplotlib.use("Agg")

# Define colors for different indicators
INDICATOR_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FECA57",  # Yellow
    "#FF9FF3",  # Pink
    "#54A0FF",  # Light Blue
    "#5F27CD",  # Purple
    "#00D2D3",  # Cyan
    "#FF9F43",  # Orange
]


def add_watermark(fig, watermark_text: str = "@amirambit", **kwargs) -> None:
    """
    Add a watermark to a matplotlib figure.

    Args:
        fig: The matplotlib figure object
        watermark_text: The text to display as watermark
        **kwargs: Additional watermark styling options including:
            - x: x position (default: 0.5)
            - y: y position (default: 0.5)
            - fontsize: font size (default: 20)
            - color: text color (default: 'gray')
            - alpha: transparency (default: 0.3)
            - rotation: rotation angle in degrees (default: 45)
            - ha: horizontal alignment (default: 'center')
            - va: vertical alignment (default: 'center')
            - zorder: drawing order (default: 1000)
    """
    # Default watermark styling
    watermark_config = {
        "x": 0.55,
        "y": 0.55,
        "fontsize": 80,
        "color": "gray",
        "alpha": 0.2,
        "rotation": 0,
        "ha": "center",
        "va": "center",
        "zorder": 1000,  # High zorder to ensure it's on top
        "transform": fig.transFigure,  # Use figure coordinates
    }

    # Update with any provided kwargs
    watermark_config.update(kwargs)

    # Add the watermark text to the figure
    fig.text(
        watermark_config["x"],
        watermark_config["y"],
        watermark_text,
        fontsize=watermark_config["fontsize"],
        color=watermark_config["color"],
        alpha=watermark_config["alpha"],
        rotation=watermark_config["rotation"],
        ha=watermark_config["ha"],
        va=watermark_config["va"],
        zorder=watermark_config["zorder"],
        transform=watermark_config["transform"],
    )

    logger.info(f"Added watermark '{watermark_text}' to chart")


def generate_ohlcv_chart(
    df: pd.DataFrame,
    indicators_to_plot: Optional[List[str]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
) -> str:
    """
    Generate an OHLCV chart with candlesticks and optional technical indicators.

    This function creates a professional-looking candlestick chart using mplfinance,
    which is specifically designed for financial data visualization. The chart includes:
    - Candlestick price data (OHLC)
    - Volume bars below the price chart
    - Optional technical indicators overlay with legend
    - Optional watermark

    Args:
        df: DataFrame with OHLCV data and datetime index
            Must have columns: Open, High, Low, Close, Volume
        indicators_to_plot: Optional list of indicator column names to overlay
                           (e.g., ['EMA_20', 'EMA_50', 'RSI_14'])
        chart_style: Style for the chart ('yahoo', 'charles', 'tradingview', etc.)
        figsize: Tuple specifying the figure size (width, height)
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options

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
        legend_labels = []

        if indicators_to_plot:
            logger.info(f"Adding indicators to chart: {indicators_to_plot}")

            for i, indicator in enumerate(indicators_to_plot):
                if indicator in df.columns:
                    # Check if the indicator has valid data
                    if not df[indicator].isna().all():
                        # Get color for this indicator
                        color = INDICATOR_COLORS[i % len(INDICATOR_COLORS)]

                        # Create addplot for each indicator with specific color
                        addplot_list.append(
                            mpf.make_addplot(
                                df[indicator],
                                type="line",
                                width=2,
                                alpha=0.8,
                                color=color,
                            )
                        )

                        # Store label and color for legend
                        legend_labels.append((indicator, color))
                        logger.info(
                            f"Added indicator {indicator} to chart with color {color}"
                        )
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

        # Add legend for indicators if any were plotted
        if legend_labels:
            # Get the main price axis (usually the first one)
            price_ax = axes[0] if isinstance(axes, (list, tuple)) else axes

            # Create legend entries
            legend_handles = []
            legend_names = []

            for label, color in legend_labels:
                # Create a line for the legend
                handle = Line2D([0], [0], color=color, linewidth=2, alpha=0.8)
                legend_handles.append(handle)
                legend_names.append(label)

            # Add the legend to the price chart
            price_ax.legend(
                legend_handles,
                legend_names,
                loc="upper left",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=10,
            )

            logger.info(f"Added legend with {len(legend_labels)} indicators")

        # Add watermark if requested
        if add_watermark_flag:
            watermark_kwargs = watermark_config or {}
            add_watermark(fig, watermark_text, **watermark_kwargs)

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
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
) -> str:
    """
    Generate an OHLCV chart with Bollinger Bands overlay.

    This is a convenience function specifically for displaying Bollinger Bands,
    which are commonly used in technical analysis. The chart will include a legend
    to identify the upper, middle, and lower Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data and datetime index
        bb_period: Period for Bollinger Bands calculation
        bb_std: Standard deviation multiplier for Bollinger Bands
        chart_style: Style for the chart
        figsize: Figure size
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options

    Returns:
        Base64 encoded string representation of the chart image with legend
    """
    # Look for Bollinger Band columns in the DataFrame
    bb_upper_col = f"BBU_{bb_period}_{bb_std}"
    bb_middle_col = f"BBM_{bb_period}_{bb_std}"
    bb_lower_col = f"BBL_{bb_period}_{bb_std}"

    indicators: List[str] = []
    # Order them for better legend display (upper, middle, lower)
    for col in [bb_upper_col, bb_middle_col, bb_lower_col]:
        if col in df.columns:
            indicators.append(col)

    if not indicators:
        logger.warning("No Bollinger Band columns found, generating chart without them")

    return generate_ohlcv_chart(
        df=df,
        indicators_to_plot=indicators,
        chart_style=chart_style,
        figsize=figsize,
        add_watermark_flag=add_watermark_flag,
        watermark_text=watermark_text,
        watermark_config=watermark_config,
    )


def generate_ohlcv_chart_with_emas(
    df: pd.DataFrame,
    ema_periods: Optional[List[int]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
) -> str:
    """
    Generate an OHLCV chart with EMA overlays.

    This function creates a chart with multiple EMA lines and includes a legend
    to identify each EMA period clearly.

    Args:
        df: DataFrame with OHLCV data and datetime index
        ema_periods: List of EMA periods to include (e.g., [20, 50, 200])
        chart_style: Style for the chart
        figsize: Figure size
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options

    Returns:
        Base64 encoded string representation of the chart image with legend
    """
    if ema_periods is None:
        ema_periods = [20, 50, 200]  # Default EMAs

    # Look for EMA columns in the DataFrame and sort by period for better legend order
    ema_indicators = []
    for period in sorted(ema_periods):  # Sort to maintain consistent order
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
        add_watermark_flag=add_watermark_flag,
        watermark_text=watermark_text,
        watermark_config=watermark_config,
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
