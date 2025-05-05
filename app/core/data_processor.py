import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, cast

import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

from app.config import INDICATOR_SETTINGS  # Import settings

# Set up logging
logger = logging.getLogger(__name__)


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

        logger.info(f"Processed {len(df)} data points into DataFrame")

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
    formatted_data = f"OHLCV Data"
    if timeframe_description:
        formatted_data += f" ({timeframe_description} timeframe)"
    formatted_data += ":\n"

    formatted_data += f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<15}\n"

    for idx, row in recent_data.iterrows():
        date_str = (
            cast(datetime, idx).strftime("%Y-%m-%d")
            if hasattr(idx, "strftime")
            else str(idx)
        )
        formatted_data += f"{date_str:<12} {row['Open']:<10.4f} {row['High']:<10.4f} "
        formatted_data += (
            f"{row['Low']:<10.4f} {row['Close']:<10.4f} {row['Volume']:<15.2f}\n"
        )

    # Add a simple summary of the full dataset
    formatted_data += f"\nSummary Statistics:\n"

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

    formatted_data += f"Latest Close: {df['Close'].iloc[-1]:.4f}\n"
    formatted_data += f"Period High: {df['High'].max():.4f}\n"
    formatted_data += f"Period Low: {df['Low'].min():.4f}\n"

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
        logger.info(f"Using custom indicator settings: {settings}")
    else:
        logger.info(f"Using default indicator settings: {INDICATOR_SETTINGS}")

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
        logger.info(f"Added technical indicators: {sorted(list(added_cols))}")

        return df_with_indicators

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # Raise a more specific error if possible, otherwise generic RuntimeError
        raise RuntimeError(f"Failed to calculate technical indicators: {e}")


def identify_support_resistance(
    df: DataFrame,
    lookback_periods: int = 20,
    cluster_threshold_percent: float = 0.5,
    min_touches: int = 2,
    top_n_levels: int = 5,
) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels using a hybrid approach.

    Args:
        df: DataFrame with OHLCV data
        lookback_periods: Window size for identifying swing points
        cluster_threshold_percent: Percentage range for clustering nearby levels
        min_touches: Minimum number of touches required to qualify as S/R
        top_n_levels: Number of top levels to return for each type

    Returns:
        Dictionary with lists of support and resistance levels

    Raises:
        ValueError: If input DataFrame is empty or insufficient data
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Cannot identify S/R levels.")
        raise ValueError("Input DataFrame is empty")

    if len(df) < lookback_periods * 2:
        logger.warning(
            f"Insufficient data points ({len(df)}) for lookback period ({lookback_periods})"
        )
        raise ValueError(
            f"Insufficient data for S/R identification (need at least {lookback_periods*2} points)"
        )

    try:
        # Step 1: Identify swing highs and lows
        logger.info(
            f"Identifying swing points with lookback period of {lookback_periods}"
        )
        df_copy = df.copy()

        # Initialize lists to store potential levels
        swing_highs = []
        swing_lows = []

        # Calculate rolling max and min within the lookback window
        df_copy["rolling_high"] = (
            df_copy["High"].rolling(window=lookback_periods, center=True).max()
        )
        df_copy["rolling_low"] = (
            df_copy["Low"].rolling(window=lookback_periods, center=True).min()
        )

        # Find points where current high/low matches the rolling max/min
        for i in range(lookback_periods, len(df_copy) - lookback_periods):
            # Check for swing high (current high is local maximum)
            if df_copy["High"].iloc[i] == df_copy["rolling_high"].iloc[i]:
                if all(
                    df_copy["High"].iloc[i] >= df_copy["High"].iloc[i - j]
                    for j in range(1, lookback_periods + 1)
                ) and all(
                    df_copy["High"].iloc[i] >= df_copy["High"].iloc[i + j]
                    for j in range(1, lookback_periods + 1)
                ):
                    swing_highs.append(
                        (i, df_copy["High"].iloc[i])
                    )  # Store index position instead of timestamp

            # Check for swing low (current low is local minimum)
            if df_copy["Low"].iloc[i] == df_copy["rolling_low"].iloc[i]:
                if all(
                    df_copy["Low"].iloc[i] <= df_copy["Low"].iloc[i - j]
                    for j in range(1, lookback_periods + 1)
                ) and all(
                    df_copy["Low"].iloc[i] <= df_copy["Low"].iloc[i + j]
                    for j in range(1, lookback_periods + 1)
                ):
                    swing_lows.append(
                        (i, df_copy["Low"].iloc[i])
                    )  # Store index position instead of timestamp

        logger.info(
            f"Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows"
        )

        # Step 2: Cluster nearby levels
        def cluster_price_levels(
            levels: List[tuple], threshold_percent: float
        ) -> List[tuple]:
            """Cluster nearby price levels and calculate importance"""
            if not levels:
                return []

            # Extract price values and calculate threshold
            prices = [level[1] for level in levels]
            avg_price = sum(prices) / len(prices)
            threshold = avg_price * threshold_percent / 100

            # Sort by price value
            sorted_levels = sorted(levels, key=lambda x: x[1])
            sorted_prices = [level[1] for level in sorted_levels]

            # Cluster nearby levels
            clusters = []
            current_cluster = [sorted_levels[0]]

            for i in range(1, len(sorted_levels)):
                if abs(sorted_prices[i] - sorted_prices[i - 1]) < threshold:
                    current_cluster.append(sorted_levels[i])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [sorted_levels[i]]

            if current_cluster:
                clusters.append(current_cluster)

            # Calculate average price and importance for each cluster
            cluster_levels = []
            for cluster in clusters:
                # Skip clusters with fewer touches than minimum
                if len(cluster) < min_touches:
                    continue

                # Extract values from the cluster
                positions = [level[0] for level in cluster]  # Integer positions
                prices = [level[1] for level in cluster]  # Price values

                # Calculate average price (the S/R level)
                avg_price = sum(prices) / len(prices)

                # Importance factors
                num_touches = len(cluster)  # Number of touches

                # Calculate recency score using integer positions directly
                # Higher index = more recent
                recency_score = sum([(pos / len(df)) for pos in positions]) / len(
                    positions
                )

                # Combined importance score (higher is more important)
                importance = num_touches * (1 + recency_score)

                cluster_levels.append((avg_price, importance))

            return cluster_levels

        # Apply clustering
        resistance_clusters = cluster_price_levels(
            swing_highs, cluster_threshold_percent
        )
        support_clusters = cluster_price_levels(swing_lows, cluster_threshold_percent)

        # Step 3: Sort by importance and select top N levels
        resistance_levels = sorted(
            resistance_clusters, key=lambda x: x[1], reverse=True
        )[:top_n_levels]
        support_levels = sorted(support_clusters, key=lambda x: x[1], reverse=True)[
            :top_n_levels
        ]

        # Step 4: Return final results (price levels only)
        result = {
            "resistance": [round(level[0], 4) for level in resistance_levels],
            "support": [round(level[0], 4) for level in support_levels],
        }

        logger.info(
            f"Identified {len(result['resistance'])} resistance and {len(result['support'])} support levels"
        )
        return result

    except Exception as e:
        logger.error(f"Error identifying support/resistance levels: {e}", exc_info=True)
        raise RuntimeError(f"Failed to identify support/resistance levels: {e}")
