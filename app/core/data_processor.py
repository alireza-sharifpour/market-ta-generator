import logging
from datetime import datetime
from typing import List, Union, cast

import pandas as pd
from pandas import DataFrame

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


def format_data_for_llm(df: DataFrame) -> str:
    """
    Format the processed DataFrame into a string suitable for an LLM prompt.

    Args:
        df: Processed DataFrame with OHLCV data.

    Returns:
        String representation of the recent data in a format suitable for LLM consumption.
    """
    # Get the latest 20 rows (or fewer if the DataFrame has fewer rows)
    recent_data = df.tail(20)

    # Create a nicely formatted string with the data
    formatted_data = "Recent OHLCV Data:\n"
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
    formatted_data += f"Latest Close: {df['Close'].iloc[-1]:.4f}\n"
    formatted_data += f"Period High: {df['High'].max():.4f}\n"
    formatted_data += f"Period Low: {df['Low'].min():.4f}\n"

    return formatted_data
