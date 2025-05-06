import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta

# Add the app directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.data_processor import prepare_llm_input_phase2


def create_test_dataframe():
    """Create a test DataFrame with OHLCV data and indicators."""
    # Create date range for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Create sample OHLCV data
    data = {
        "Open": [100.0 + i * 0.5 for i in range(len(dates))],
        "High": [105.0 + i * 0.5 for i in range(len(dates))],
        "Low": [95.0 + i * 0.5 for i in range(len(dates))],
        "Close": [102.0 + i * 0.5 for i in range(len(dates))],
        "Volume": [1000000.0 + i * 10000 for i in range(len(dates))],
    }

    # Create DataFrame with the date index
    df = pd.DataFrame(data, index=dates)

    # Calculate indicators using pandas_ta
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.mfi(length=14, append=True)

    return df


def test_prepare_llm_input_phase2():
    """Test the prepare_llm_input_phase2 function."""
    # Create test DataFrame with indicators
    df = create_test_dataframe()

    # Create sample support/resistance levels
    sr_levels = {"support": [95.0, 97.5, 100.0], "resistance": [110.0, 115.0, 120.0]}

    # Call the function
    result = prepare_llm_input_phase2(df, sr_levels)

    # Print the result
    print("=== Test Result: prepare_llm_input_phase2 ===")
    print(result)
    print("============================================")

    # Basic validation
    assert isinstance(result, str), "Result should be a string"
    assert "MARKET DATA ANALYSIS" in result, "Result should contain title"
    assert "Latest OHLCV Data" in result, "Result should contain OHLCV section"
    assert "Technical Indicators" in result, "Result should contain indicators section"
    assert (
        "Support and Resistance Levels" in result
    ), "Result should contain S/R section"

    print("All assertions passed!")


if __name__ == "__main__":
    test_prepare_llm_input_phase2()
