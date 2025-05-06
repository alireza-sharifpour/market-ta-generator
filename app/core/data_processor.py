import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, cast

import numpy as np
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


def prepare_llm_input_phase2(
    df_with_indicators: DataFrame, sr_levels: Dict[str, List[float]]
) -> str:
    """
    Extract key information from the indicator-enriched DataFrame and S/R levels
    to prepare a structured input for the Phase 2 LLM prompt.

    Args:
        df_with_indicators: DataFrame with OHLCV data and calculated technical indicators
        sr_levels: Dictionary with lists of support and resistance levels

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

    # OHLCV summary
    output += f"## Latest OHLCV Data ({date_str})\n"
    output += f"- Open: {latest['Open']:.4f}\n"
    output += f"- High: {latest['High']:.4f}\n"
    output += f"- Low: {latest['Low']:.4f}\n"
    output += f"- Close: {latest['Close']:.4f}\n"
    output += f"- Volume: {latest['Volume']:.2f}\n\n"

    # Key statistics
    output += "## Market Statistics\n"
    output += f"- Period High (last {len(df_with_indicators)} periods): {df_with_indicators['High'].max():.4f}\n"
    output += f"- Period Low (last {len(df_with_indicators)} periods): {df_with_indicators['Low'].min():.4f}\n"

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
            output += f"- EMA_{period_str}: {value:.4f}{trend}\n"
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
            band_width_value = (
                float(latest[bbwidth_col]) * 100
            )  # Convert to percentage and ensure float type
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

        output += f"- Bollinger Upper ({bb_period},{bb_std}): {bb_upper:.4f} ({upper_distance:.2f}% above price)\n"
        output += f"- Bollinger Middle ({bb_period},{bb_std}): {bb_middle:.4f}\n"
        output += f"- Bollinger Lower ({bb_period},{bb_std}): {bb_lower:.4f} ({lower_distance:.2f}% below price)\n"
        output += f"- Band Width: {band_width_value:.2f}% of price{position}\n"
    else:
        logger.warning(
            f"Bollinger Bands columns not found in DataFrame. Expected format: BBU/BBL/BBM_{bb_period}_{bb_std}."
        )

    # Support and Resistance Levels
    output += "\n## Support and Resistance Levels\n"

    # Current price for reference
    current_price = latest["Close"]

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
                output += f"- {level:.4f} ({distance:.2f}% above current price)\n"
        else:
            output += "- No resistance levels detected above current price\n"

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
                output += f"- {level:.4f} ({distance:.2f}% below current price)\n"
        else:
            output += "- No support levels detected below current price\n"

    return output
