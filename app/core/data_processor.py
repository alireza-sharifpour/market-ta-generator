import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import pandas_ta  # type: ignore  # noqa: F401  # Required for DataFrame.ta extension
from pandas import DataFrame

from app.config import INDICATOR_SETTINGS, SR_SETTINGS  # Import settings

# Set up logging
logger = logging.getLogger(__name__)


def format_price_smart(price: float) -> str:
    """
    Format price values with appropriate decimal precision based on magnitude.

    Args:
        price: The price value to format

    Returns:
        Formatted price string with appropriate decimal places
    """
    if price == 0:
        return "0.0000"

    abs_price = abs(price)

    if abs_price >= 1:
        # For prices >= 1, use 4 decimal places
        return f"{price:.4f}"
    elif abs_price >= 0.01:
        # For prices >= 0.01, use 6 decimal places
        return f"{price:.6f}"
    elif abs_price >= 0.0001:
        # For prices >= 0.0001, use 8 decimal places
        return f"{price:.8f}"
    else:
        # For very small prices, use scientific notation or many decimal places
        formatted = f"{price:.10f}".rstrip("0").rstrip(".")
        if len(formatted.split(".")[-1]) > 10:
            return f"{price:.2e}"
        return formatted


def process_raw_data(raw_data: List[List[Union[int, float]]]) -> DataFrame:
    """
    Convert raw OHLCV data from LBank API into a pandas DataFrame.

    Args:
        raw_data: List of lists with format [timestamp, open, high, low, close, volume]
                 where timestamp is in seconds since epoch.

    Returns:
        pandas DataFrame with columns [Open, High, Low, Close, Volume] and
        a datetime index converted from the timestamp.

    Raises:
        ValueError: If the input data is empty or not in the expected format.
    """
    # Validate input data
    if not raw_data:
        raise ValueError("Raw data cannot be empty")

    try:
        # Create DataFrame with appropriate column names
        df = pd.DataFrame(
            raw_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        )

        # Convert timestamp (seconds since epoch) to datetime objects
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

        # Set timestamp as index
        df.set_index("Timestamp", inplace=True)

        # Sort by timestamp in ascending order (oldest first)
        df.sort_index(inplace=True)

        logger.debug(f"Processed {len(df)} data points into DataFrame")

        return df

    except Exception as e:
        logger.error(f"Error processing raw data: {str(e)}")
        raise ValueError(f"Failed to process data: {str(e)}")


def format_data_for_llm(df: DataFrame, timeframe: Optional[str] = None) -> str:
    """
    Format the processed DataFrame into a string suitable for an LLM prompt.

    Args:
        df: Processed DataFrame with OHLCV data.
        timeframe: Time interval for each candle (e.g., "day1", "hour4")

    Returns:
        String representation of all the data in a format suitable for LLM consumption.
    """
    # Use all rows instead of just the last 20
    recent_data = df

    # Add timeframe information if provided
    timeframe_description = ""
    if timeframe:
        if timeframe == "minute1":
            timeframe_description = "1-minute"
        elif timeframe == "minute5":
            timeframe_description = "5-minute"
        elif timeframe == "minute15":
            timeframe_description = "15-minute"
        elif timeframe == "minute30":
            timeframe_description = "30-minute"
        elif timeframe == "hour1":
            timeframe_description = "1-hour"
        elif timeframe == "hour4":
            timeframe_description = "4-hour"
        elif timeframe == "hour8":
            timeframe_description = "8-hour"
        elif timeframe == "hour12":
            timeframe_description = "12-hour"
        elif timeframe == "day1":
            timeframe_description = "daily"
        elif timeframe == "week1":
            timeframe_description = "weekly"
        elif timeframe == "month1":
            timeframe_description = "monthly"
        else:
            timeframe_description = timeframe

    # Create a nicely formatted string with the data
    formatted_data = "OHLCV Data"
    if timeframe_description:
        formatted_data += f" ({timeframe_description} timeframe)"
    formatted_data += ":\n"

    formatted_data += f"{'Date':<12} {'Open':<15} {'High':<15} {'Low':<15} {'Close':<15} {'Volume':<15}\n"

    for idx, row in recent_data.iterrows():
        date_str = (
            cast(datetime, idx).strftime("%Y-%m-%d")
            if hasattr(idx, "strftime")
            else str(idx)
        )
        open_str = format_price_smart(row["Open"])
        high_str = format_price_smart(row["High"])
        low_str = format_price_smart(row["Low"])
        close_str = format_price_smart(row["Close"])
        formatted_data += f"{date_str:<12} {open_str:<15} {high_str:<15} "
        formatted_data += f"{low_str:<15} {close_str:<15} {row['Volume']:<15.2f}\n"

    # Add a simple summary of the full dataset
    formatted_data += "\nSummary Statistics:\n"

    # Safely format datetime index values
    min_date = df.index.min()
    max_date = df.index.max()
    min_date_str = (
        min_date.strftime("%Y-%m-%d")
        if hasattr(min_date, "strftime")
        else str(min_date)
    )
    max_date_str = (
        max_date.strftime("%Y-%m-%d")
        if hasattr(max_date, "strftime")
        else str(max_date)
    )

    formatted_data += f"Data Range: {min_date_str} to {max_date_str}\n"
    formatted_data += f"Total Periods: {len(df)}\n"

    if timeframe_description:
        formatted_data += f"Timeframe: {timeframe_description}\n"

    formatted_data += f"Latest Close: {format_price_smart(df['Close'].iloc[-1])}\n"
    formatted_data += f"Period High: {format_price_smart(df['High'].max())}\n"
    formatted_data += f"Period Low: {format_price_smart(df['Low'].min())}\n"

    return formatted_data


def add_technical_indicators(
    df: DataFrame, settings: Optional[Dict[str, Union[int, float]]] = None
) -> DataFrame:
    """
    Calculate technical indicators for the OHLCV DataFrame using pandas_ta.

    Uses default periods from app.config.INDICATOR_SETTINGS if not provided.
    Required indicators: EMAs, RSI, Bollinger Bands, ADX, MFI.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume]
            and a datetime index.
        settings: Optional dictionary to override default indicator parameters.
                  Keys should match INDICATOR_SETTINGS keys.

    Returns:
        DataFrame with added technical indicator columns.

    Raises:
        ValueError: If the input DataFrame is empty or lacks required columns.
        RuntimeError: If an error occurs during indicator calculation.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot calculate indicators.")
        raise ValueError("Input DataFrame is empty")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"DataFrame missing required columns: {missing}")
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Use default settings, override with provided settings
    current_settings = INDICATOR_SETTINGS.copy()
    if settings:
        current_settings.update(settings)
        logger.debug(f"Using custom indicator settings: {settings}")
    else:
        logger.debug(f"Using default indicator settings: {INDICATOR_SETTINGS}")

    # Create a copy to avoid modifying the original DataFrame
    df_with_indicators = df.copy()

    try:
        # Ensure required periods are available for calculation
        min_data_points = max(
            current_settings["ema_long"],
            current_settings["rsi_period"],
            current_settings["bb_period"],
            current_settings["adx_period"],
            current_settings["mfi_period"],
        )
        if len(df_with_indicators) < min_data_points:
            logger.warning(
                f"Insufficient data points ({len(df_with_indicators)}) for the longest period ({min_data_points}). Indicators may be NaN."
            )
            # Do not raise error here, let pandas-ta handle NaN generation

        # Add indicators using the DataFrame extension provided by pandas_ta
        df_with_indicators.ta.ema(length=current_settings["ema_short"], append=True)
        df_with_indicators.ta.ema(length=current_settings["ema_medium"], append=True)
        df_with_indicators.ta.ema(length=current_settings["ema_long"], append=True)
        df_with_indicators.ta.rsi(length=current_settings["rsi_period"], append=True)
        df_with_indicators.ta.bbands(
            length=current_settings["bb_period"],
            std=current_settings["bb_std_dev"],
            append=True,
        )
        df_with_indicators.ta.adx(length=current_settings["adx_period"], append=True)
        df_with_indicators.ta.mfi(length=current_settings["mfi_period"], append=True)

        # Log the columns added
        added_cols = set(df_with_indicators.columns) - set(df.columns)
        logger.debug(f"Added technical indicators: {sorted(list(added_cols))}")

        return df_with_indicators

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # Raise a more specific error if possible, otherwise generic RuntimeError
        raise RuntimeError(f"Failed to calculate technical indicators: {e}")


def get_sr_settings_for_timeframe(timeframe: Optional[str] = None) -> Dict[str, Any]:
    """
    Get appropriate S/R settings based on timeframe.

    Args:
        timeframe: Timeframe string (e.g., "hour1", "minute5", "day1")

    Returns:
        Dictionary with S/R settings appropriate for the timeframe
    """
    if not timeframe:
        return SR_SETTINGS["default"]  # type: ignore

    # Check which category the timeframe belongs to
    for category, settings in SR_SETTINGS.items():
        if category == "default":
            continue
        if isinstance(settings, dict) and timeframe in settings.get("timeframes", []):
            return settings

    # If timeframe not found, return default settings
    logger.warning(f"Timeframe '{timeframe}' not found in SR_SETTINGS, using default")
    return SR_SETTINGS["default"]  # type: ignore


def identify_support_resistance(
    df: DataFrame,
    timeframe: Optional[str] = None,
    lookback_periods: Optional[int] = None,
    cluster_threshold_percent: Optional[float] = None,
    min_touches: Optional[int] = None,
    top_n_levels: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels using multiple detection methods.

    This enhanced version uses:
    1. Swing point detection (traditional method)
    2. Density-based level detection (for horizontal levels)
    3. Timeframe-specific parameter adjustment
    4. Improved clustering algorithms

    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe string for parameter optimization (e.g., "hour1", "minute5")
        lookback_periods: Window size for swing point detection (overrides timeframe default)
        cluster_threshold_percent: Percentage range for clustering levels (overrides timeframe default)
        min_touches: Minimum touches required to qualify as S/R (overrides timeframe default)
        top_n_levels: Number of top levels to return for each type (overrides timeframe default)

    Returns:
        Dictionary with lists of support and resistance levels

    Raises:
        ValueError: If input DataFrame is empty or insufficient data
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot identify S/R levels.")
        raise ValueError("Input DataFrame is empty")

    # Get timeframe-specific settings
    sr_settings = get_sr_settings_for_timeframe(timeframe)

    # Use provided parameters or fall back to timeframe-specific defaults
    final_lookback_periods = (
        lookback_periods
        if lookback_periods is not None
        else sr_settings["lookback_periods"]
    )
    final_cluster_threshold_percent = (
        cluster_threshold_percent
        if cluster_threshold_percent is not None
        else sr_settings["cluster_threshold_percent"]
    )
    final_min_touches = (
        min_touches if min_touches is not None else sr_settings["min_touches"]
    )
    final_top_n_levels = (
        top_n_levels if top_n_levels is not None else sr_settings["top_n_levels"]
    )
    density_window = sr_settings["density_window"]
    volatility_adjustment = sr_settings["volatility_adjustment"]

    if len(df) < final_lookback_periods * 2:
        logger.warning(
            f"Insufficient data points ({len(df)}) for lookback period ({final_lookback_periods})"
        )
        raise ValueError(
            f"Insufficient data for S/R identification (need at least {final_lookback_periods*2} points)"
        )

    try:
        logger.info(
            f"Identifying S/R levels with timeframe: {timeframe}, lookback: {final_lookback_periods}, "
            f"cluster_threshold: {final_cluster_threshold_percent}%, min_touches: {final_min_touches}"
        )

        df_copy = df.copy()
        current_price = df["Close"].iloc[-1]

        # Step 1: Swing Point Detection (Traditional Method)
        swing_highs, swing_lows = _detect_swing_points(df_copy, final_lookback_periods)

        # Step 2: Density-Based Level Detection (For Horizontal Levels)
        density_highs, density_lows = _detect_density_levels(
            df_copy, density_window, volatility_adjustment
        )

        # Step 3: Combine Detection Methods
        all_highs = swing_highs + density_highs
        all_lows = swing_lows + density_lows

        logger.info(
            f"Found {len(swing_highs)} swing highs, {len(density_highs)} density highs, "
            f"{len(swing_lows)} swing lows, {len(density_lows)} density lows"
        )

        # Step 4: Enhanced Clustering
        resistance_clusters = _cluster_price_levels(
            all_highs, final_cluster_threshold_percent, final_min_touches, df_copy
        )
        support_clusters = _cluster_price_levels(
            all_lows, final_cluster_threshold_percent, final_min_touches, df_copy
        )

        # Step 5: Sort by importance and select top N levels
        resistance_levels = sorted(
            resistance_clusters, key=lambda x: x[1], reverse=True
        )[:final_top_n_levels]
        support_levels = sorted(support_clusters, key=lambda x: x[1], reverse=True)[
            :final_top_n_levels
        ]

        # Step 6: Dynamic reclassification based on current price
        final_resistance, final_support = _reclassify_levels(
            resistance_levels, support_levels, current_price
        )

        result = {
            "resistance": final_resistance,
            "support": final_support,
        }

        logger.info(
            f"Identified {len(result['resistance'])} resistance and {len(result['support'])} support levels "
            f"after dynamic reclassification (current price: {current_price:.4f})"
        )

        return result

    except Exception as e:
        logger.error(f"Error identifying support/resistance levels: {e}", exc_info=True)
        raise RuntimeError(f"Failed to identify support/resistance levels: {e}")


def _detect_swing_points(
    df: DataFrame, lookback_periods: int
) -> tuple[List[tuple], List[tuple]]:
    """
    Detect swing highs and lows using traditional swing point method.

    Args:
        df: DataFrame with OHLCV data
        lookback_periods: Window size for swing point detection

    Returns:
        Tuple of (swing_highs, swing_lows) where each is a list of (index, price) tuples
    """
    swing_highs = []
    swing_lows = []
    window_half = lookback_periods // 2

    for i in range(window_half, len(df) - window_half):
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]

        # Check for swing high
        is_swing_high = True
        for j in range(max(0, i - window_half), min(len(df), i + window_half + 1)):
            if j != i and df["High"].iloc[j] >= current_high:
                is_swing_high = False
                break

        if is_swing_high:
            swing_highs.append((i, current_high))

        # Check for swing low
        is_swing_low = True
        for j in range(max(0, i - window_half), min(len(df), i + window_half + 1)):
            if j != i and df["Low"].iloc[j] <= current_low:
                is_swing_low = False
                break

        if is_swing_low:
            swing_lows.append((i, current_low))

    return swing_highs, swing_lows


def _detect_density_levels(
    df: DataFrame, density_window: int, volatility_adjustment: bool
) -> tuple[List[tuple], List[tuple]]:
    """
    Detect support/resistance levels using price density analysis.
    This method finds horizontal levels where price spent significant time.

    Args:
        df: DataFrame with OHLCV data
        density_window: Window size for density calculation
        volatility_adjustment: Whether to adjust thresholds based on volatility

    Returns:
        Tuple of (density_highs, density_lows) where each is a list of (index, price) tuples
    """
    density_highs: List[tuple] = []
    density_lows: List[tuple] = []

    if len(df) < density_window * 2:
        return density_highs, density_lows

    # Calculate price ranges for density analysis
    price_range = df["High"].max() - df["Low"].min()

    # Adjust density threshold based on volatility if enabled
    if volatility_adjustment:
        recent_volatility = df["Close"].pct_change().iloc[-20:].std()
        density_threshold = price_range * (
            0.01 + recent_volatility * 0.5
        )  # Adaptive threshold
    else:
        density_threshold = price_range * 0.015  # Fixed 1.5% threshold

    # Find areas with high price density (multiple touches)
    for i in range(density_window, len(df) - density_window):
        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]

        # Count nearby price touches for highs
        high_touches = 0
        for j in range(
            max(0, i - density_window), min(len(df), i + density_window + 1)
        ):
            if abs(df["High"].iloc[j] - current_high) <= density_threshold:
                high_touches += 1

        # Count nearby price touches for lows
        low_touches = 0
        for j in range(
            max(0, i - density_window), min(len(df), i + density_window + 1)
        ):
            if abs(df["Low"].iloc[j] - current_low) <= density_threshold:
                low_touches += 1

        # If we have enough touches, consider it a significant level
        if high_touches >= 3:  # Minimum 3 touches for density level
            density_highs.append((i, current_high))

        if low_touches >= 3:  # Minimum 3 touches for density level
            density_lows.append((i, current_low))

    # Remove duplicates by clustering very close levels
    density_highs = _remove_duplicate_levels(density_highs, density_threshold)
    density_lows = _remove_duplicate_levels(density_lows, density_threshold)

    return density_highs, density_lows


def _remove_duplicate_levels(levels: List[tuple], threshold: float) -> List[tuple]:
    """
    Remove duplicate levels that are very close to each other.

    Args:
        levels: List of (index, price) tuples
        threshold: Price threshold for considering levels as duplicates

    Returns:
        List of unique levels
    """
    if not levels:
        return []

    # Sort by price
    sorted_levels = sorted(levels, key=lambda x: x[1])
    unique_levels = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        # If this level is far enough from the last unique level, keep it
        if abs(level[1] - unique_levels[-1][1]) > threshold:
            unique_levels.append(level)

    return unique_levels


def _cluster_price_levels(
    levels: List[tuple], threshold_percent: float, min_touches: int, df: DataFrame
) -> List[tuple]:
    """
    Enhanced clustering of price levels with importance calculation.

    Args:
        levels: List of (index, price) tuples
        threshold_percent: Percentage range for clustering nearby levels
        min_touches: Minimum number of touches required to qualify as S/R
        df: DataFrame for calculating importance metrics

    Returns:
        List of (price, importance) tuples
    """
    if not levels:
        return []

    # Extract price values and calculate threshold
    prices = [level[1] for level in levels]
    avg_price = sum(prices) / len(prices)
    threshold = avg_price * threshold_percent / 100

    # Sort by price value
    sorted_levels = sorted(levels, key=lambda x: x[1])

    # Cluster nearby levels
    clusters = []
    current_cluster = [sorted_levels[0]]

    for i in range(1, len(sorted_levels)):
        if abs(sorted_levels[i][1] - sorted_levels[i - 1][1]) < threshold:
            current_cluster.append(sorted_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_levels[i]]

    if current_cluster:
        clusters.append(current_cluster)

    # Calculate average price and importance for each cluster
    cluster_levels = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue

        # Extract values from the cluster
        positions = [level[0] for level in cluster]
        prices = [level[1] for level in cluster]

        # Calculate average price (the S/R level)
        avg_price = sum(prices) / len(prices)

        # Enhanced importance calculation
        num_touches = len(cluster)

        # Recency score (higher index = more recent)
        recency_score = sum([pos / len(df) for pos in positions]) / len(positions)

        # Volume-weighted importance (if we have volume data)
        volume_weight = 1.0
        try:
            # Get volume data for the positions in this cluster
            cluster_volumes = [df["Volume"].iloc[pos] for pos in positions]
            avg_volume = sum(cluster_volumes) / len(cluster_volumes)
            overall_avg_volume = df["Volume"].mean()
            volume_weight = min(2.0, avg_volume / overall_avg_volume)  # Cap at 2x
        except (IndexError, KeyError, ValueError):
            pass  # If volume calculation fails, use default weight

        # Combined importance score
        importance = num_touches * (1 + recency_score) * volume_weight

        cluster_levels.append((avg_price, importance))

    return cluster_levels


def _reclassify_levels(
    resistance_levels: List[tuple], support_levels: List[tuple], current_price: float
) -> tuple[List[float], List[float]]:
    """
    Dynamically reclassify support/resistance levels based on current price.

    Args:
        resistance_levels: List of (price, importance) tuples for resistance
        support_levels: List of (price, importance) tuples for support
        current_price: Current market price

    Returns:
        Tuple of (final_resistance, final_support) lists
    """
    # Extract price values
    all_resistance_levels = [round(level[0], 4) for level in resistance_levels]
    all_support_levels = [round(level[0], 4) for level in support_levels]

    # Reclassify based on current price
    final_resistance = []
    final_support = []

    # Process original resistance levels
    for level in all_resistance_levels:
        if level > current_price:
            final_resistance.append(level)  # Still resistance (above price)
        else:
            final_support.append(level)  # Now support (price broke above)

    # Process original support levels
    for level in all_support_levels:
        if level < current_price:
            final_support.append(level)  # Still support (below price)
        else:
            final_resistance.append(level)  # Now resistance (price broke below)

    # Remove duplicates and sort
    final_resistance = sorted(list(set(final_resistance)))
    final_support = sorted(list(set(final_support)), reverse=True)

    return final_resistance, final_support


def should_reclassify_sr_levels(
    original_price: float, 
    current_price: float, 
    threshold: Optional[float] = None
) -> bool:
    """
    Check if S/R levels need reclassification based on price movement.
    
    Args:
        original_price: Price used for original S/R calculation
        current_price: Current market price
        threshold: Price movement threshold (defaults to config value)
        
    Returns:
        True if reclassification needed, False otherwise
    """
    from app.config import SR_RECLASSIFICATION_THRESHOLD
    
    if threshold is None:
        threshold = SR_RECLASSIFICATION_THRESHOLD
    
    if original_price <= 0:
        logger.warning(f"Invalid original price: {original_price}")
        return False
        
    price_change_pct = abs((current_price - original_price) / original_price)
    
    if price_change_pct > threshold:
        logger.info(
            f"Significant price movement detected: {price_change_pct:.2%} "
            f"(threshold: {threshold:.2%}). S/R reclassification needed."
        )
        return True
    
    return False


def reclassify_cached_sr_levels(
    cached_sr_levels: Dict[str, List[float]], 
    current_price: float
) -> Dict[str, List[float]]:
    """
    Reclassify cached S/R levels using current price.
    
    Args:
        cached_sr_levels: Dictionary with 'support' and 'resistance' level lists
        current_price: Current market price for reclassification
        
    Returns:
        Dictionary with reclassified S/R levels
        
    Raises:
        ValueError: If cached S/R levels are invalid
    """
    try:
        if not cached_sr_levels or not isinstance(cached_sr_levels, dict):
            raise ValueError("Invalid cached S/R levels format")
            
        # Get original levels
        original_resistance = cached_sr_levels.get("resistance", [])
        original_support = cached_sr_levels.get("support", [])
        
        if not original_resistance and not original_support:
            logger.warning("No S/R levels found in cache to reclassify")
            return cached_sr_levels
        
        # Convert to format expected by _reclassify_levels function
        # _reclassify_levels expects (price, importance) tuples, but we only have prices
        resistance_tuples = [(price, 1.0) for price in original_resistance]
        support_tuples = [(price, 1.0) for price in original_support]
        
        # Use existing reclassification logic
        reclassified_resistance, reclassified_support = _reclassify_levels(
            resistance_tuples, support_tuples, current_price
        )
        
        result = {
            "resistance": reclassified_resistance,
            "support": reclassified_support,
        }
        
        logger.info(
            f"Reclassified S/R levels with current price {current_price:.4f}: "
            f"{len(result['resistance'])} resistance, {len(result['support'])} support"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error reclassifying S/R levels: {e}")
        # Return original levels on error to avoid breaking the analysis
        return cached_sr_levels


def prepare_llm_input_phase2(
    df_with_indicators: DataFrame,
    sr_levels: Dict[str, List[float]],
    current_price_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Extract key information from the indicator-enriched DataFrame and S/R levels
    to prepare a structured input for the Phase 2 LLM prompt.

    Args:
        df_with_indicators: DataFrame with OHLCV data and calculated technical indicators
        sr_levels: Dictionary with lists of support and resistance levels
        current_price_data: Optional dictionary with current price data from ticker API

    Returns:
        Structured string with key market data, indicator values, trends, and S/R levels
        suitable for the advanced LLM prompt.

    Raises:
        ValueError: If the input DataFrame is empty or missing critical indicators
    """
    if df_with_indicators.empty:
        raise ValueError("Input DataFrame is empty")

    # Get the most recent data point (latest values)
    latest = df_with_indicators.iloc[-1]

    # Get the previous data point for trend comparison
    prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else None

    # Extract OHLCV data for the latest period
    latest_date = df_with_indicators.index[-1]
    date_str = (
        latest_date.strftime("%Y-%m-%d %H:%M")
        if hasattr(latest_date, "strftime")
        else str(latest_date)
    )

    # Prepare structured output
    output = "# MARKET DATA ANALYSIS\n\n"

    # Add current price information if available
    if current_price_data:
        output += "## Current Market Price (Live)\n"
        ticker = current_price_data.get("ticker", {})
        if ticker:
            current_price = ticker.get("latest")
            if current_price:
                output += (
                    f"- Current Price: {format_price_smart(float(current_price))}\n"
                )

                # Calculate difference from last close
                price_diff = float(current_price) - latest["Close"]
                price_diff_pct = (price_diff / latest["Close"]) * 100

                output += f"- Change from Last Close: {format_price_smart(price_diff)} ({price_diff_pct:+.2f}%)\n"

                # Add 24hr statistics if available
                high_24h = ticker.get("high")
                low_24h = ticker.get("low")
                vol_24h = ticker.get("vol")

                if high_24h:
                    output += f"- 24h High: {format_price_smart(float(high_24h))}\n"
                if low_24h:
                    output += f"- 24h Low: {format_price_smart(float(low_24h))}\n"
                if vol_24h:
                    output += f"- 24h Volume: {float(vol_24h):.2f}\n"
        output += "\n"

    # OHLCV summary
    output += f"## Latest OHLCV Data ({date_str})\n"
    output += f"- Open: {format_price_smart(latest['Open'])}\n"
    output += f"- High: {format_price_smart(latest['High'])}\n"
    output += f"- Low: {format_price_smart(latest['Low'])}\n"
    output += f"- Close: {format_price_smart(latest['Close'])}\n"
    output += f"- Volume: {latest['Volume']:.2f}\n\n"

    # Key statistics
    output += "## Market Statistics\n"
    output += f"- Period High (last {len(df_with_indicators)} periods): {format_price_smart(df_with_indicators['High'].max())}\n"
    output += f"- Period Low (last {len(df_with_indicators)} periods): {format_price_smart(df_with_indicators['Low'].min())}\n"

    # Calculate recent price change and volatility
    if len(df_with_indicators) > 1:
        price_change = (
            (latest["Close"] - df_with_indicators.iloc[-2]["Close"])
            / df_with_indicators.iloc[-2]["Close"]
            * 100
        )
        output += f"- Last Period Change: {price_change:.2f}%\n"

        # Calculate 5-period volatility (standard deviation of returns)
        if len(df_with_indicators) >= 5:
            volatility = df_with_indicators["Close"].pct_change().iloc[-5:].std() * 100
            output += f"- Recent Volatility (5-period): {volatility:.2f}%\n\n"
        else:
            output += "\n"
    else:
        output += "\n"

    # Technical Indicators section
    output += "## Technical Indicators\n"

    # Get indicator periods from INDICATOR_SETTINGS
    # from app.config import INDICATOR_SETTINGS # Removed redundant import

    # Check for and add EMA values (using exact column names from settings)
    ema_periods = [
        INDICATOR_SETTINGS["ema_short"],
        INDICATOR_SETTINGS["ema_medium"],
        INDICATOR_SETTINGS["ema_long"],
    ]

    ema_columns = []
    for period in ema_periods:
        ema_col = f"EMA_{period}"
        if ema_col in df_with_indicators.columns:
            ema_columns.append(ema_col)

    if ema_columns:
        output += "### Moving Averages\n"
        for col in sorted(ema_columns, key=lambda x: int(x.split("_")[1])):
            # Extract period from column name (e.g., "EMA_10" -> "10")
            period_str = col.split("_")[1]  # Keep as string to avoid type errors

            # Get the value as a scalar using numpy's item() method which is safer
            value = np.asarray(latest[col]).item()

            # Determine trend by comparing with previous value
            trend = ""
            if prev is not None:
                # Get scalar values for both current and previous
                current_val = np.asarray(latest[col]).item()
                prev_val = np.asarray(prev[col]).item()

                # Now compare the scalar values
                if current_val > prev_val:
                    trend = " (Rising)"
                elif current_val < prev_val:
                    trend = " (Falling)"
                else:
                    trend = " (Neutral)"
            output += f"- EMA_{period_str}: {format_price_smart(value)}{trend}\n"
    else:
        logger.warning(
            "No EMA columns found in DataFrame. Expected EMA_{period} format."
        )

    # RSI with period
    rsi_period = INDICATOR_SETTINGS["rsi_period"]
    rsi_col = f"RSI_{rsi_period}"
    if rsi_col in df_with_indicators.columns:
        output += "\n### Momentum Indicators\n"
        rsi_value = float(latest[rsi_col])  # Convert to float for scalar value
        rsi_trend = ""
        if prev is not None:
            # Extract values as scalars for comparison
            current_rsi = float(latest[rsi_col])
            prev_rsi = float(prev[rsi_col])
            if current_rsi > prev_rsi:
                rsi_trend = " (Rising)"
            elif current_rsi < prev_rsi:
                rsi_trend = " (Falling)"
            else:
                rsi_trend = " (Neutral)"

        # Add interpretation for RSI
        rsi_interpretation = ""
        if rsi_value >= 70:
            rsi_interpretation = " - Potentially Overbought"
        elif rsi_value <= 30:
            rsi_interpretation = " - Potentially Oversold"

        output += (
            f"- RSI_{rsi_period}: {rsi_value:.2f}{rsi_trend}{rsi_interpretation}\n"
        )
    else:
        logger.warning(f"RSI column (RSI_{rsi_period}) not found in DataFrame.")

    # MFI with period
    mfi_period = INDICATOR_SETTINGS["mfi_period"]
    mfi_col = f"MFI_{mfi_period}"
    if mfi_col in df_with_indicators.columns:
        mfi_value = float(latest[mfi_col])  # Convert to float for scalar value
        mfi_trend = ""
        if prev is not None:
            # Extract values as scalars for comparison
            current_mfi = float(latest[mfi_col])
            prev_mfi = float(prev[mfi_col])
            if current_mfi > prev_mfi:
                mfi_trend = " (Rising)"
            elif current_mfi < prev_mfi:
                mfi_trend = " (Falling)"
            else:
                mfi_trend = " (Neutral)"

        # Add interpretation for MFI
        mfi_interpretation = ""
        if mfi_value >= 80:
            mfi_interpretation = " - Potentially Overbought"
        elif mfi_value <= 20:
            mfi_interpretation = " - Potentially Oversold"

        output += (
            f"- MFI_{mfi_period}: {mfi_value:.2f}{mfi_trend}{mfi_interpretation}\n"
        )
    else:
        logger.warning(f"MFI column (MFI_{mfi_period}) not found in DataFrame.")

    # ADX with period
    adx_period = INDICATOR_SETTINGS["adx_period"]
    adx_col = f"ADX_{adx_period}"
    if adx_col in df_with_indicators.columns:
        output += "\n### Trend Strength\n"
        adx_value = float(latest[adx_col])  # Convert to float for scalar value

        # Add interpretation for ADX
        trend_strength = ""
        if adx_value >= 25:
            trend_strength = " - Strong Trend"
        elif adx_value < 20:
            trend_strength = " - Weak or No Trend"
        else:
            trend_strength = " - Moderate Trend"

        output += f"- ADX_{adx_period}: {adx_value:.2f}{trend_strength}\n"

        # Add DI+ and DI- with period information
        dmp_col = f"DMP_{adx_period}"
        dmn_col = f"DMN_{adx_period}"

        if (
            dmp_col in df_with_indicators.columns
            and dmn_col in df_with_indicators.columns
        ):
            dmp_value = float(latest[dmp_col])  # Convert to float for scalar value
            dmn_value = float(latest[dmn_col])  # Convert to float for scalar value
            trend_direction = ""
            if dmp_value > dmn_value:
                trend_direction = " - Bullish Trend Direction"
            elif dmp_value < dmn_value:
                trend_direction = " - Bearish Trend Direction"

            output += f"- DI+_{adx_period}: {dmp_value:.2f}\n"
            output += f"- DI-_{adx_period}: {dmn_value:.2f}{trend_direction}\n"
        else:
            logger.warning(
                f"DMP/DMN columns (DMP_{adx_period}/DMN_{adx_period}) not found in DataFrame."
            )
    else:
        logger.warning(f"ADX column (ADX_{adx_period}) not found in DataFrame.")

    # Bollinger Bands with period and std
    bb_period = INDICATOR_SETTINGS["bb_period"]
    bb_std = INDICATOR_SETTINGS["bb_std_dev"]

    # Define standard pandas-ta BB column names
    bbupper_col = f"BBU_{bb_period}_{bb_std}"
    bbmiddle_col = f"BBM_{bb_period}_{bb_std}"
    bblower_col = f"BBL_{bb_period}_{bb_std}"
    bbwidth_col = f"BBB_{bb_period}_{bb_std}"

    # Check if at least upper and lower bands exist
    has_upper = bbupper_col in df_with_indicators.columns
    has_lower = bblower_col in df_with_indicators.columns
    has_middle = bbmiddle_col in df_with_indicators.columns  # Check for middle band
    has_width = bbwidth_col in df_with_indicators.columns  # Check for bandwidth

    if has_upper and has_lower:
        output += "\n### Volatility Bands\n"
        bb_upper = latest[bbupper_col]
        bb_lower = latest[bblower_col]

        # Use middle band directly if available, otherwise calculate
        if has_middle:
            bb_middle = latest[bbmiddle_col]
        else:
            bb_middle = (
                np.asarray(bb_upper).item() + np.asarray(bb_lower).item()
            ) / 2  # Ensure scalar math
            logger.info(
                f"Bollinger middle band column ({bbmiddle_col}) not found. Calculating manually."
            )

        # Distance from current price to bands (as percentage)
        upper_distance = (
            (np.asarray(bb_upper).item() - latest["Close"]) / latest["Close"] * 100
        )
        lower_distance = (
            (latest["Close"] - np.asarray(bb_lower).item()) / latest["Close"] * 100
        )

        # Use bandwidth directly if available, otherwise calculate
        if has_width:
            # BBB from pandas-ta is already a percentage, no need to multiply by 100
            band_width_value = float(latest[bbwidth_col])
        else:
            # Ensure bb_middle is a scalar for division
            bb_middle_scalar = (
                np.asarray(bb_middle).item()
                if not isinstance(bb_middle, (int, float))
                else bb_middle
            )
            if bb_middle_scalar != 0:
                band_width_value = (
                    (np.asarray(bb_upper).item() - np.asarray(bb_lower).item())
                    / bb_middle_scalar
                    * 100
                )
            else:
                band_width_value = 0  # Avoid division by zero
                logger.warning(
                    f"Bollinger middle band is zero, cannot calculate bandwidth for {bbwidth_col}."
                )
            logger.info(
                f"Bollinger bandwidth column ({bbwidth_col}) not found. Calculating manually."
            )

        # Price position relative to bands
        position = ""

        # Convert pandas Series to numpy arrays and extract scalar values
        close_val = np.asarray(latest["Close"]).item()
        bb_upper_val = np.asarray(bb_upper).item()
        bb_lower_val = np.asarray(bb_lower).item()

        # Compare scalar values
        if close_val > bb_upper_val:
            position = " - Price above upper band (potential overbought)"
        elif close_val < bb_lower_val:
            position = " - Price below lower band (potential oversold)"

        output += f"- Bollinger Upper ({bb_period},{bb_std}): {format_price_smart(bb_upper)} ({upper_distance:.2f}% above price)\n"
        output += f"- Bollinger Middle ({bb_period},{bb_std}): {format_price_smart(bb_middle)}\n"
        output += f"- Bollinger Lower ({bb_period},{bb_std}): {format_price_smart(bb_lower)} ({lower_distance:.2f}% below price)\n"
        output += f"- Band Width: {band_width_value:.2f}% of price{position}\n"
    else:
        logger.warning(
            f"Bollinger Bands columns not found in DataFrame. Expected format: BBU/BBL/BBM_{bb_period}_{bb_std}."
        )

    # Support and Resistance Levels
    output += "\n## Support and Resistance Levels\n"

    # Current price for reference - use live price if available, otherwise use last close
    if current_price_data and current_price_data.get("ticker", {}).get("latest"):
        current_price = float(current_price_data["ticker"]["latest"])
        output += f"*Using live current price: {format_price_smart(current_price)}*\n\n"
    else:
        current_price = latest["Close"]
        output += f"*Using last close price: {format_price_smart(current_price)}*\n\n"

    # Process resistance levels (sort in ascending order)
    if "resistance" in sr_levels and sr_levels["resistance"]:
        output += "### Resistance Levels (Ascending)\n"
        # Sort resistance levels above current price in ascending order
        resistance_levels = sorted(
            [level for level in sr_levels["resistance"] if level > current_price]
        )
        if resistance_levels:
            for level in resistance_levels:
                distance = ((level - current_price) / current_price) * 100
                output += f"- {format_price_smart(level)} ({distance:.2f}% above current price)\n"
        else:
            # This case handles when sr_levels["resistance"] was not empty,
            # but all identified levels were below the current price.
            output += "- No resistance levels above current price (all identified levels are below current price).\n"
    else:  # If "resistance" key not in sr_levels or sr_levels["resistance"] is an empty list
        output += "### Resistance Levels (Ascending)\n"  # Still add the heading
        output += "- No significant resistance levels detected with current settings.\n"

    # Process support levels (sort in descending order)
    if "support" in sr_levels and sr_levels["support"]:
        output += "\n### Support Levels (Descending)\n"
        # Sort support levels below current price in descending order
        support_levels = sorted(
            [level for level in sr_levels["support"] if level < current_price],
            reverse=True,
        )
        if support_levels:
            for level in support_levels:
                distance = ((current_price - level) / current_price) * 100
                output += f"- {format_price_smart(level)} ({distance:.2f}% below current price)\n"
        else:
            # This case handles when sr_levels["support"] was not empty,
            # but all identified levels were above the current price.
            output += "- No support levels below current price (all identified levels are above current price).\n"
    else:  # If "support" key not in sr_levels or sr_levels["support"] is an empty list
        output += "\n### Support Levels (Descending)\n"  # Still add the heading
        output += "- No significant support levels detected with current settings.\n"

    return output


def prepare_llm_input_for_cache(
    df_with_indicators: DataFrame,
    sr_levels: Dict[str, List[float]],
) -> str:
    """
    Extract key information from the indicator-enriched DataFrame and S/R levels
    to prepare a structured input for caching. This version EXCLUDES current price data
    to ensure cache key stability.

    Args:
        df_with_indicators: DataFrame with OHLCV data and calculated technical indicators
        sr_levels: Dictionary with lists of support and resistance levels

    Returns:
        Structured string with key market data, indicator values, trends, and S/R levels
        suitable for LLM caching (without current price data).

    Raises:
        ValueError: If the input DataFrame is empty or missing critical indicators
    """
    if df_with_indicators.empty:
        raise ValueError("Input DataFrame is empty")

    # Get the most recent data point (latest values)
    latest = df_with_indicators.iloc[-1]

    # Get the previous data point for trend comparison
    prev = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else None

    # Extract OHLCV data for the latest period
    latest_date = df_with_indicators.index[-1]
    date_str = (
        latest_date.strftime("%Y-%m-%d %H:%M")
        if hasattr(latest_date, "strftime")
        else str(latest_date)
    )

    # Prepare structured output
    output = "# MARKET DATA ANALYSIS\n\n"

    # NOTE: Current price section is EXCLUDED for cache stability

    # OHLCV summary
    output += f"## Latest OHLCV Data ({date_str})\n"
    output += f"- Open: {format_price_smart(latest['Open'])}\n"
    output += f"- High: {format_price_smart(latest['High'])}\n"
    output += f"- Low: {format_price_smart(latest['Low'])}\n"
    output += f"- Close: {format_price_smart(latest['Close'])}\n"
    output += f"- Volume: {latest['Volume']:.2f}\n\n"

    # Key statistics
    output += "## Market Statistics\n"
    output += f"- Period High (last {len(df_with_indicators)} periods): {format_price_smart(df_with_indicators['High'].max())}\n"
    output += f"- Period Low (last {len(df_with_indicators)} periods): {format_price_smart(df_with_indicators['Low'].min())}\n"

    # Calculate recent price change and volatility
    if len(df_with_indicators) > 1:
        price_change = (
            (latest["Close"] - df_with_indicators.iloc[-2]["Close"])
            / df_with_indicators.iloc[-2]["Close"]
            * 100
        )
        output += f"- Last Period Change: {price_change:.2f}%\n"

        # Calculate 5-period volatility (standard deviation of returns)
        if len(df_with_indicators) >= 5:
            volatility = df_with_indicators["Close"].pct_change().iloc[-5:].std() * 100
            output += f"- Recent Volatility (5-period): {volatility:.2f}%\n\n"
        else:
            output += "\n"
    else:
        output += "\n"

    # Technical Indicators section (same as original)
    output += "## Technical Indicators\n"

    # Check for and add EMA values
    ema_periods = [
        INDICATOR_SETTINGS["ema_short"],
        INDICATOR_SETTINGS["ema_medium"],
        INDICATOR_SETTINGS["ema_long"],
    ]

    ema_columns = []
    for period in ema_periods:
        ema_col = f"EMA_{period}"
        if ema_col in df_with_indicators.columns:
            ema_columns.append(ema_col)

    if ema_columns:
        output += "### Moving Averages\n"
        for col in sorted(ema_columns, key=lambda x: int(x.split("_")[1])):
            period_str = col.split("_")[1]
            value = np.asarray(latest[col]).item()

            trend = ""
            if prev is not None:
                current_val = np.asarray(latest[col]).item()
                prev_val = np.asarray(prev[col]).item()

                if current_val > prev_val:
                    trend = " (Rising)"
                elif current_val < prev_val:
                    trend = " (Falling)"
                else:
                    trend = " (Neutral)"
            output += f"- EMA_{period_str}: {format_price_smart(value)}{trend}\n"

    # RSI with period
    rsi_period = INDICATOR_SETTINGS["rsi_period"]
    rsi_col = f"RSI_{rsi_period}"
    if rsi_col in df_with_indicators.columns:
        output += "\n### Momentum Indicators\n"
        rsi_value = float(latest[rsi_col])
        rsi_trend = ""
        if prev is not None:
            current_rsi = float(latest[rsi_col])
            prev_rsi = float(prev[rsi_col])
            if current_rsi > prev_rsi:
                rsi_trend = " (Rising)"
            elif current_rsi < prev_rsi:
                rsi_trend = " (Falling)"
            else:
                rsi_trend = " (Neutral)"

        rsi_interpretation = ""
        if rsi_value >= 70:
            rsi_interpretation = " - Potentially Overbought"
        elif rsi_value <= 30:
            rsi_interpretation = " - Potentially Oversold"

        output += (
            f"- RSI_{rsi_period}: {rsi_value:.2f}{rsi_trend}{rsi_interpretation}\n"
        )

    # MFI with period
    mfi_period = INDICATOR_SETTINGS["mfi_period"]
    mfi_col = f"MFI_{mfi_period}"
    if mfi_col in df_with_indicators.columns:
        mfi_value = float(latest[mfi_col])
        mfi_trend = ""
        if prev is not None:
            current_mfi = float(latest[mfi_col])
            prev_mfi = float(prev[mfi_col])
            if current_mfi > prev_mfi:
                mfi_trend = " (Rising)"
            elif current_mfi < prev_mfi:
                mfi_trend = " (Falling)"
            else:
                mfi_trend = " (Neutral)"

        mfi_interpretation = ""
        if mfi_value >= 80:
            mfi_interpretation = " - Potentially Overbought"
        elif mfi_value <= 20:
            mfi_interpretation = " - Potentially Oversold"

        output += (
            f"- MFI_{mfi_period}: {mfi_value:.2f}{mfi_trend}{mfi_interpretation}\n"
        )

    # ADX and Directional Indicators
    adx_period = INDICATOR_SETTINGS["adx_period"]
    adx_col = f"ADX_{adx_period}"
    di_plus_col = f"DMP_{adx_period}"
    di_minus_col = f"DMN_{adx_period}"

    if all(
        col in df_with_indicators.columns
        for col in [adx_col, di_plus_col, di_minus_col]
    ):
        output += "\n### Trend Strength Indicators\n"

        adx_value = float(latest[adx_col])
        di_plus_value = float(latest[di_plus_col])
        di_minus_value = float(latest[di_minus_col])

        trend_strength = ""
        if adx_value >= 50:
            trend_strength = " - Very Strong Trend"
        elif adx_value >= 25:
            trend_strength = " - Strong Trend"
        elif adx_value >= 20:
            trend_strength = " - Moderate Trend"
        else:
            trend_strength = " - Weak/No Trend"

        output += f"- ADX_{adx_period}: {adx_value:.2f}{trend_strength}\n"
        output += f"- DI+_{adx_period}: {di_plus_value:.2f}\n"
        output += f"- DI-_{adx_period}: {di_minus_value:.2f}\n"

        if di_plus_value > di_minus_value:
            output += "- Directional Movement: Bullish (DI+ > DI-)\n"
        elif di_minus_value > di_plus_value:
            output += "- Directional Movement: Bearish (DI- > DI+)\n"
        else:
            output += "- Directional Movement: Neutral (DI+ ≈ DI-)\n"

    # Bollinger Bands
    bb_period = INDICATOR_SETTINGS["bb_period"]
    bb_std = INDICATOR_SETTINGS["bb_std_dev"]
    bb_upper_col = f"BBU_{bb_period}_{bb_std}"
    bb_middle_col = f"BBM_{bb_period}_{bb_std}"
    bb_lower_col = f"BBL_{bb_period}_{bb_std}"

    if all(
        col in df_with_indicators.columns
        for col in [bb_upper_col, bb_middle_col, bb_lower_col]
    ):
        output += "\n### Volatility Indicators (Bollinger Bands)\n"

        bb_upper = float(latest[bb_upper_col])
        bb_middle = float(latest[bb_middle_col])
        bb_lower = float(latest[bb_lower_col])
        current_close = latest["Close"]

        output += (
            f"- BB Upper ({bb_period}, {bb_std}): {format_price_smart(bb_upper)}\n"
        )
        output += (
            f"- BB Middle ({bb_period}, {bb_std}): {format_price_smart(bb_middle)}\n"
        )
        output += (
            f"- BB Lower ({bb_period}, {bb_std}): {format_price_smart(bb_lower)}\n"
        )

        # Position relative to bands
        if current_close >= bb_upper:
            position = "Above Upper Band - Potentially Overbought"
        elif current_close <= bb_lower:
            position = "Below Lower Band - Potentially Oversold"
        elif current_close >= bb_middle:
            position = "Above Middle Band - Upper Half"
        else:
            position = "Below Middle Band - Lower Half"

        output += f"- Current Position: {position}\n"

        # Band width
        band_width = ((bb_upper - bb_lower) / bb_middle) * 100
        output += f"- Band Width: {band_width:.2f}% (Volatility measure)\n"

    # Support and Resistance Levels
    output += "\n## Support and Resistance Levels\n"

    # Use last close price for reference (since we exclude current price)
    current_price = latest["Close"]
    output += f"*Using last close price: {format_price_smart(current_price)}*\n\n"

    # Process resistance levels (sort in ascending order)
    if "resistance" in sr_levels and sr_levels["resistance"]:
        output += "### Resistance Levels (Ascending)\n"
        resistance_levels = sorted(
            [level for level in sr_levels["resistance"] if level > current_price]
        )
        if resistance_levels:
            for level in resistance_levels:
                distance = ((level - current_price) / current_price) * 100
                output += f"- {format_price_smart(level)} ({distance:.2f}% above current price)\n"
        else:
            output += "- No resistance levels detected above current price with current settings.\n"
    else:
        output += "### Resistance Levels (Ascending)\n"
        output += "- No significant resistance levels detected with current settings.\n"

    # Process support levels (sort in descending order)
    if "support" in sr_levels and sr_levels["support"]:
        output += "\n### Support Levels (Descending)\n"
        support_levels = sorted(
            [level for level in sr_levels["support"] if level < current_price],
            reverse=True,
        )
        if support_levels:
            for level in support_levels:
                distance = ((current_price - level) / current_price) * 100
                output += f"- {format_price_smart(level)} ({distance:.2f}% below current price)\n"
        else:
            output += "- No support levels detected below current price with current settings.\n"
    else:
        output += "\n### Support Levels (Descending)\n"
        output += "- No significant support levels detected with current settings.\n"

    return output
