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


def filter_chart_indicators(df: pd.DataFrame) -> List[str]:
    """
    Filter available indicators to show only EMA50 and EMA9 on charts.

    This function is used to limit chart display to specific indicators while
    keeping all calculated indicators available for LLM analysis and reporting.

    Args:
        df: DataFrame with OHLCV data and technical indicators

    Returns:
        List of indicator column names to display on charts (EMA50 and EMA9 only)
    """
    chart_indicators = []

    # Define the specific indicators to show on charts
    target_indicators = ["EMA_9", "EMA_50"]

    for indicator in target_indicators:
        if indicator in df.columns:
            # Check if the indicator has valid data
            if not df[indicator].isna().all():
                chart_indicators.append(indicator)
                logger.info(f"Added {indicator} to chart display")
            else:
                logger.warning(f"Indicator {indicator} has no valid data, skipping")
        else:
            logger.warning(f"Indicator {indicator} not found in DataFrame")

    logger.info(f"Filtered chart indicators: {chart_indicators}")
    return chart_indicators


def generate_ohlcv_chart(
    df: pd.DataFrame,
    indicators_to_plot: Optional[List[str]] = None,
    sr_levels: Optional[Dict[str, List[float]]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
    use_filtered_indicators: bool = True,
) -> str:
    """
    Generate an OHLCV chart with candlesticks and optional technical indicators.

    This function creates a professional-looking candlestick chart using mplfinance,
    which is specifically designed for financial data visualization. The chart includes:
    - Candlestick price data (OHLC)
    - Volume bars below the price chart
    - Optional technical indicators overlay with legend
    - Optional Support and Resistance levels as horizontal lines
    - Optional watermark

    Args:
        df: DataFrame with OHLCV data and datetime index
            Must have columns: Open, High, Low, Close, Volume
        indicators_to_plot: Optional list of indicator column names to overlay
                           (e.g., ['EMA_20', 'EMA_50', 'RSI_14'])
                           If None and use_filtered_indicators is True, will use filtered indicators
        sr_levels: Optional dictionary with support and resistance levels
                  Expected format: {'support': [list of levels], 'resistance': [list of levels]}
        chart_style: Style for the chart ('yahoo', 'charles', 'tradingview', etc.)
        figsize: Tuple specifying the figure size (width, height)
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options
        use_filtered_indicators: If True and indicators_to_plot is None, use only EMA50 and EMA9

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

    # Determine which indicators to plot
    if indicators_to_plot is None and use_filtered_indicators:
        indicators_to_plot = filter_chart_indicators(df)
        logger.info("Using filtered indicators for chart display")
    elif indicators_to_plot is None:
        indicators_to_plot = []
        logger.info("No indicators specified for chart display")

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

        # Get the main price axis for adding S/R levels and legend
        price_ax = axes[0] if isinstance(axes, (list, tuple)) else axes

        # Add Support and Resistance levels if provided
        if sr_levels:
            logger.info("Adding Support and Resistance levels to chart")

            # Get current price range for better visualization
            price_min = df["Low"].min()
            price_max = df["High"].max()

            # Add resistance levels (red horizontal lines)
            if "resistance" in sr_levels and sr_levels["resistance"]:
                for level in sr_levels["resistance"]:
                    # Only show levels within reasonable range of current data
                    if price_min * 0.8 <= level <= price_max * 1.2:
                        price_ax.axhline(
                            y=level,
                            color="red",
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.7,
                            zorder=10,
                        )
                        # Add text label for the level
                        price_ax.text(
                            len(df) * 0.02,  # Position near left side
                            level,
                            f"R: {level:.4f}",
                            color="red",
                            fontsize=8,
                            alpha=0.8,
                            verticalalignment="bottom",
                            zorder=20,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                            ),
                        )

                # Add resistance to legend
                legend_labels.append(("Resistance", "red"))
                logger.info(
                    f"Added {len([l for l in sr_levels['resistance'] if price_min * 0.8 <= l <= price_max * 1.2])} resistance levels"
                )

            # Add support levels (green horizontal lines)
            if "support" in sr_levels and sr_levels["support"]:
                for level in sr_levels["support"]:
                    # Only show levels within reasonable range of current data
                    if price_min * 0.8 <= level <= price_max * 1.2:
                        price_ax.axhline(
                            y=level,
                            color="green",
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.7,
                            zorder=10,
                        )
                        # Add text label for the level
                        price_ax.text(
                            len(df) * 0.02,  # Position near left side
                            level,
                            f"S: {level:.4f}",
                            color="green",
                            fontsize=8,
                            alpha=0.8,
                            verticalalignment="top",
                            zorder=20,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                            ),
                        )

                # Add support to legend
                legend_labels.append(("Support", "green"))
                logger.info(
                    f"Added {len([l for l in sr_levels['support'] if price_min * 0.8 <= l <= price_max * 1.2])} support levels"
                )

        # Add legend for indicators and S/R levels if any were plotted
        if legend_labels:
            # Create legend entries
            legend_handles = []
            legend_names = []

            for label, color in legend_labels:
                if label in ["Support", "Resistance"]:
                    # Create dashed line for S/R levels
                    handle = Line2D(
                        [0], [0], color=color, linestyle="--", linewidth=1.5, alpha=0.7
                    )
                else:
                    # Create solid line for indicators
                    handle = Line2D([0], [0], color=color, linewidth=2, alpha=0.8)
                legend_handles.append(handle)
                legend_names.append(label)

            # Add the legend to the price chart
            legend = price_ax.legend(
                legend_handles,
                legend_names,
                loc="upper left",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=10,
            )
            # Set the legend to appear above all other elements
            legend.set_zorder(30)

            logger.info(f"Added legend with {len(legend_labels)} items")

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
    sr_levels: Optional[Dict[str, List[float]]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
    use_filtered_indicators: bool = True,
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
        sr_levels: Optional dictionary with support and resistance levels
        chart_style: Style for the chart
        figsize: Figure size
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options
        use_filtered_indicators: If True, use only EMA50 and EMA9 instead of Bollinger Bands

    Returns:
        Base64 encoded string representation of the chart image with legend
    """
    # If using filtered indicators, prioritize EMA50 and EMA9 over Bollinger Bands
    if use_filtered_indicators:
        return generate_ohlcv_chart(
            df=df,
            indicators_to_plot=None,  # Will use filter_chart_indicators
            sr_levels=sr_levels,
            chart_style=chart_style,
            figsize=figsize,
            add_watermark_flag=add_watermark_flag,
            watermark_text=watermark_text,
            watermark_config=watermark_config,
            use_filtered_indicators=True,
        )

    # Original Bollinger Bands logic (when use_filtered_indicators=False)
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
        sr_levels=sr_levels,
        chart_style=chart_style,
        figsize=figsize,
        add_watermark_flag=add_watermark_flag,
        watermark_text=watermark_text,
        watermark_config=watermark_config,
        use_filtered_indicators=False,  # Don't double-filter
    )


def generate_ohlcv_chart_with_emas(
    df: pd.DataFrame,
    ema_periods: Optional[List[int]] = None,
    sr_levels: Optional[Dict[str, List[float]]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
    use_filtered_indicators: bool = True,
) -> str:
    """
    Generate an OHLCV chart with EMA overlays.

    This function creates a chart with multiple EMA lines and includes a legend
    to identify each EMA period clearly.

    Args:
        df: DataFrame with OHLCV data and datetime index
        ema_periods: List of EMA periods to include (e.g., [20, 50, 200])
        sr_levels: Optional dictionary with support and resistance levels
        chart_style: Style for the chart
        figsize: Figure size
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options
        use_filtered_indicators: If True, use only EMA50 and EMA9 regardless of ema_periods

    Returns:
        Base64 encoded string representation of the chart image with legend
    """
    # If using filtered indicators, use EMA50 and EMA9 regardless of ema_periods parameter
    if use_filtered_indicators:
        return generate_ohlcv_chart(
            df=df,
            indicators_to_plot=None,  # Will use filter_chart_indicators
            sr_levels=sr_levels,
            chart_style=chart_style,
            figsize=figsize,
            add_watermark_flag=add_watermark_flag,
            watermark_text=watermark_text,
            watermark_config=watermark_config,
            use_filtered_indicators=True,
        )

    # Original EMA logic (when use_filtered_indicators=False)
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
        sr_levels=sr_levels,
        chart_style=chart_style,
        figsize=figsize,
        add_watermark_flag=add_watermark_flag,
        watermark_text=watermark_text,
        watermark_config=watermark_config,
        use_filtered_indicators=False,  # Don't double-filter
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


def generate_ohlcv_chart_with_support_resistance(
    df: pd.DataFrame,
    sr_levels: Dict[str, List[float]],
    indicators_to_plot: Optional[List[str]] = None,
    chart_style: str = "yahoo",
    figsize: tuple = (12, 8),
    add_watermark_flag: bool = True,
    watermark_text: str = "@amirambit",
    watermark_config: Optional[Dict] = None,
    use_filtered_indicators: bool = True,
) -> str:
    """
    Generate an OHLCV chart with prominent Support and Resistance level display.

    This is a convenience function specifically designed to highlight Support and
    Resistance levels on the chart. It provides a clean interface for displaying
    S/R levels with optional technical indicators.

    Args:
        df: DataFrame with OHLCV data and datetime index
        sr_levels: Dictionary with support and resistance levels
                  Required format: {'support': [list of levels], 'resistance': [list of levels]}
        indicators_to_plot: Optional list of indicator column names to overlay
        chart_style: Style for the chart
        figsize: Figure size
        add_watermark_flag: Whether to add a watermark to the chart
        watermark_text: Text to display as watermark
        watermark_config: Optional dictionary with watermark styling options
        use_filtered_indicators: If True and indicators_to_plot is None, use only EMA50 and EMA9

    Returns:
        Base64 encoded string representation of the chart image with S/R levels highlighted

    Raises:
        ValueError: If sr_levels is None or empty
    """
    if not sr_levels:
        logger.error("Support and Resistance levels are required for this chart type")
        raise ValueError("sr_levels parameter is required and cannot be empty")

    # Validate sr_levels structure
    if not isinstance(sr_levels, dict):
        raise ValueError("sr_levels must be a dictionary")

    if "support" not in sr_levels and "resistance" not in sr_levels:
        raise ValueError("sr_levels must contain 'support' and/or 'resistance' keys")

    logger.info("Generating OHLCV chart with highlighted Support and Resistance levels")

    return generate_ohlcv_chart(
        df=df,
        indicators_to_plot=indicators_to_plot,
        sr_levels=sr_levels,
        chart_style=chart_style,
        figsize=figsize,
        add_watermark_flag=add_watermark_flag,
        watermark_text=watermark_text,
        watermark_config=watermark_config,
        use_filtered_indicators=use_filtered_indicators,
    )
