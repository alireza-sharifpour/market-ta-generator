"""
LLM Client module for interacting with Large Language Models.
This implementation uses OpenAI by default, but is designed to be easily swappable.
"""

import abc
import logging
import time
from typing import Any, Dict, Optional

# Import error types directly from openai package
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app.config import AVALAI_API_BASE_URL, AVALAI_API_KEY, OPENAI_MODEL

# Configure logger
logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients to enable easy provider switching."""

    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The generated text response

        Raises:
            Exception: If there's an error in text generation
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of the LLM client."""

    def __init__(self, api_key: str = AVALAI_API_KEY, model: str = OPENAI_MODEL):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini')
        """
        self.api_key = api_key
        self.model = model

        print(f"Initializing OpenAI client with model: {model}")

        if not api_key:
            logger.error("OpenAI API key not provided.")
            raise ValueError("OpenAI API key is required.")

        try:
            self.client = OpenAI(api_key=api_key, base_url=AVALAI_API_BASE_URL)
            logger.info(f"Initialized OpenAI client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def generate_text(
        self, prompt: str, max_retries: int = 3, retry_delay: int = 2
    ) -> str:
        """
        Generate text using OpenAI API.

        Args:
            prompt: The prompt to send to OpenAI
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            The generated text response

        Raises:
            Exception: If there's an error in text generation after all retries
        """
        retries = 0

        while retries <= max_retries:
            try:
                # Log the full prompt at INFO level with a distinctive format
                logger.info("======== FULL PROMPT SENT TO LLM ========")
                logger.info(prompt)
                logger.info("==========================================")

                # Make the API call to OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional cryptocurrency analyst.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,  # Lower temperature for more consistent analytical output
                )

                # Extract the generated text from the response
                if response and response.choices and len(response.choices) > 0:
                    generated_text = response.choices[0].message.content
                    if generated_text:
                        generated_text = generated_text.strip()
                        logger.debug(
                            f"Received response from OpenAI: {generated_text[:100]}..."
                        )
                        return generated_text
                    else:
                        raise ValueError("Empty response content from OpenAI")
                else:
                    raise ValueError("Invalid response structure from OpenAI")

            except (RateLimitError, APIConnectionError) as e:
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"OpenAI API error: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Max retries reached. Failed to generate text with OpenAI: {str(e)}"
                    )
                    raise

            except AuthenticationError as e:
                logger.error(f"OpenAI authentication error: {str(e)}")
                raise

            except BadRequestError as e:
                # Bad request errors are client errors and won't be fixed by retry
                logger.error(f"OpenAI bad request error: {str(e)}")
                raise

            except APIError as e:
                # For API errors like server errors, also retry
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"OpenAI API error: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})"
                    )
                    time.sleep(retry_delay * 2)  # Wait longer for API errors
                else:
                    logger.error(f"OpenAI API error after max retries: {str(e)}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error with OpenAI API: {str(e)}")
                raise

        # This should not be reached if the loop exits correctly, but adding as a failsafe
        raise RuntimeError("Failed to generate text: Maximum retries exceeded")


# Factory function to get the configured LLM client
def get_llm_client(
    provider: str = "openai", config: Optional[Dict[str, Any]] = None
) -> BaseLLMClient:
    """
    Factory function to get the configured LLM client.

    Args:
        provider: The LLM provider to use (currently only 'openai' is implemented)
        config: Configuration options for the provider

    Returns:
        An instance of the configured LLM client

    Raises:
        ValueError: If the provider is not supported
    """
    config = config or {}

    if provider.lower() == "openai":
        return OpenAIClient(**config)
    # Add support for additional providers here
    # elif provider.lower() == "anthropic":
    #    return AnthropicClient(**config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_basic_analysis(
    pair: str, data_summary: str, timeframe: Optional[str] = None
) -> str:
    """
    Generate a basic analysis of the cryptocurrency pair data.

    Args:
        pair: The trading pair (e.g., 'BTCUSDT')
        data_summary: Processed data summary from the data_processor
        timeframe: The timeframe of the data (e.g., 'day1', 'hour4')

    Returns:
        The generated analysis text

    Raises:
        Exception: If there's an error in the analysis generation
    """
    try:
        # Get human-readable timeframe description
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

        prompt = f"""
        You are a professional cryptocurrency analyst generating a report for a Telegram bot.
        **Important Note:** This is Phase 1 analysis based **strictly on the provided OHLCV data only**. It describes recent price action but **cannot provide indicator-based signals, specific scenarios, or predictions**.

        **Input Data:**
        Trading Pair: **{pair}**
        Timeframe: **{timeframe_description}** # English timeframe description (e.g., 4-hour, daily)
        Recent OHLCV Data Summary (Candlesticks):

        {data_summary} # Tabular data starts with a header row, then rows with columns: Date, Open, High, Low, Close, Volume. Dates are in YYYY-MM-DD format. PAY CLOSE ATTENTION TO THE 'Date' COLUMN.

        **Output Requirements:**
        1.  **Language:** MUST be entirely in **Persian (Farsi)**.
        2.  **Formatting:** Use **Telegram Markdown** (`**bold**`, `- ` bullets).
        3.  **Structure:**
            * **Title:** Start immediately with the Persian title, strictly following this structure: `**تحلیل {pair} - تایم فریم [PERSIAN_TIMEFRAME_PHRASE]**`.
                * Convert the Input Timeframe (`{timeframe_description}`) into the `[PERSIAN_TIMEFRAME_PHRASE]` using natural Persian TA phrasing. **Examples:**
                    * Input `daily` -> Use `روزانه` -> Full Title: `**تحلیل {pair} - تایم فریم روزانه**`
                    * Input `4-hour` -> Use `۴ ساعته` -> Full Title: `**تحلیل {pair} - تایم فریم ۴ ساعته**`
                    * Input `1-hour` -> Use `۱ ساعته` -> Full Title: `**تحلیل {pair} - تایم فریم ۱ ساعته**`
                    *(Adapt pattern for others)*
            * **Data Period Identification (Instruction for LLM):**
                * **Carefully examine** the `Date` column in the `data_summary` provided above. Ignore the header row.
                * Locate the date in the **first data row**; this is the **[START_DATE]**.
                * Locate the date in the **very last data row**; this is the **[END_DATE]**.
                * **CRITICAL:** Extract the dates exactly as they appear (YYYY-MM-DD format). **Verify the year.** For example, if the first date is `2025-04-07`, use `2025-04-07`. **Do not output incorrect years like 20025.**
            * **Body:** Follow the title (with a blank line) using these exact Persian headings:

                `**۱. خلاصه وضعیت:**`
                - Provide a brief overview for **{pair}** in the specified Persian timeframe. State that the analysis covers the period from **[START_DATE]** to **[END_DATE]** (using the exact dates identified from the first and last data rows).
                - Calculate and mention the approximate **overall percentage change** from the *start to the end* of the provided data.
                - Describe the price action in the **last 1-3 candles** within the data set.
                - Briefly comment on recent **volume** compared to the average volume in the provided data set.

                `**۲. روند و سطوح مشاهده شده:**`
                - State the primary **trend** observed *during the analyzed period* (from **[START_DATE]** to **[END_DATE]**).
                - Specify the **highest price** (`بالاترین قیمت در این دوره`) and **lowest price** (`پایین‌ترین قیمت در این دوره`) reached *within this specific period* (from **[START_DATE]** to **[END_DATE]**).
                - Report the **most recent closing price** (`آخرین قیمت بسته‌شدن`) and mention where it sits relative to the high and low *of this period*.

                `**۳. احساسات کلی (بر اساس قیمت/حجم):**`
                - Conclude the overall market sentiment derived *strictly* from the observed price/volume *in the analyzed period* (from **[START_DATE]** to **[END_DATE]**).
                # Disclaimer instruction has been removed.

        **Important Constraints:**
        * Analysis MUST be based *only* on the provided OHLCV `data_summary`.
        * **Accurately extract and use the start/end dates (YYYY-MM-DD format)** from the first and last *data rows* (ignore header) of the `Date` column. Double-check the year is correct (e.g., 2025).
        * Acknowledge this is basic analysis; **do not** invent signals, scenarios, predictions.
        * Strictly follow the requested Persian title and heading structure. Use the timeframe conversion examples.
        * No trading advice.
        * Output ONLY the Persian title and structured analysis.
        """

        # Get the LLM client (defaults to OpenAI)
        llm_client = get_llm_client()

        # Generate the analysis using the LLM
        analysis = llm_client.generate_text(prompt)

        return analysis

    except Exception as e:
        logger.error(f"Error generating analysis for {pair}: {str(e)}")
        raise


def generate_summarized_analysis(
    pair: str, structured_data: str, timeframe: Optional[str] = None
) -> str:
    """
    Generate a summarized analysis of the cryptocurrency pair,
    incorporating technical indicators and S/R levels in a short format.

    Args:
        pair: The trading pair (e.g., 'BTCUSDT')
        structured_data: A string containing formatted OHLCV data,
                         technical indicators, and S/R levels.
        timeframe: The timeframe of the data (e.g., 'day1', 'hour4')

    Returns:
        The generated summarized analysis text in Persian.

    Raises:
        Exception: If there's an error in the analysis generation.
    """
    try:
        # Get Persian timeframe phrase
        persian_timeframe_phrase = ""
        if timeframe:
            if timeframe == "minute1":
                persian_timeframe_phrase = "1M"
            elif timeframe == "minute5":
                persian_timeframe_phrase = "5M"
            elif timeframe == "minute15":
                persian_timeframe_phrase = "15M"
            elif timeframe == "minute30":
                persian_timeframe_phrase = "30M"
            elif timeframe == "hour1":
                persian_timeframe_phrase = "1H"
            elif timeframe == "hour4":
                persian_timeframe_phrase = "4H"
            elif timeframe == "hour8":
                persian_timeframe_phrase = "8H"
            elif timeframe == "hour12":
                persian_timeframe_phrase = "12H"
            elif timeframe == "day1":
                persian_timeframe_phrase = "1D"
            elif timeframe == "week1":
                persian_timeframe_phrase = "1W"
            elif timeframe == "month1":
                persian_timeframe_phrase = "1M"
            else:
                persian_timeframe_phrase = timeframe

        prompt = f"""
        You are a professional cryptocurrency analyst. Create a SHORT summarized analysis for {pair} in Persian using the provided data.

        **Input Data:**
        Trading Pair: {pair}
        Timeframe: {persian_timeframe_phrase}
        
        ```
        {structured_data}
        ```

        **Output Requirements:**
        1. Language: **Persian (Farsi) only**
        2. Format: Telegram Markdown
        3. Keep it SHORT - maximum 10-12 lines
        4. Follow this EXACT structure:

        📊 تحلیل {pair} - تایم‌فریم {persian_timeframe_phrase}

        ▫️وضعیت کلی:
        - قیمت لحظه‌ای: [current_price]
        - روند بلندمدت --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - روند کوتاه‌مدت --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - حمایت مهم بعدی: [support_level]
        - مقاومت مهم بعدی: [resistance_level]

        💭 توصیه عملی:
        - [Short practical recommendation based on indicators]
        - نقطه ورود احتمالی: [entry suggestion based on EMAs/indicators]
        - سطح کلیدی برای تایید: [confirmation level]
        - سطح ریسک: [متوسط/بالا/پایین]

        ⚠️ نکات مهم:
        - [Key warning or note about EMA levels/resistance failure]
        - [Important level that could change the outlook]

        **Important Instructions:**
        - Use ONLY the provided structured data
        - For trend strength, use exactly one of: قوی, متوسط, ضعیف
        - For trend direction, use exactly one of: صعودی, نزولی, خنثی
        - Extract current price from the latest OHLCV data
        - Use identified Support/Resistance levels from the data
        - Base EMA recommendations on the actual EMA values provided
        - Keep recommendations practical and specific
        - Output ONLY the formatted analysis - no extra text
        """

        # Get the LLM client
        llm_client = get_llm_client()

        # Generate the analysis using the LLM
        analysis = llm_client.generate_text(prompt)

        return analysis

    except Exception as e:
        logger.error(f"Error generating summarized analysis for {pair}: {str(e)}")
        raise


def generate_detailed_analysis(
    pair: str, structured_data: str, timeframe: Optional[str] = None
) -> str:
    """
    Generate a detailed (Phase 2) analysis of the cryptocurrency pair,
    incorporating technical indicators and S/R levels.

    Args:
        pair: The trading pair (e.g., 'BTCUSDT')
        structured_data: A string containing formatted OHLCV data,
                         technical indicators, and S/R levels.
        timeframe: The timeframe of the data (e.g., 'day1', 'hour4')

    Returns:
        The generated detailed analysis text in Persian.

    Raises:
        Exception: If there's an error in the analysis generation.
    """
    try:
        # Get human-readable timeframe description
        timeframe_description = ""
        persian_timeframe_phrase = ""
        if timeframe:
            if timeframe == "minute1":
                timeframe_description = "1-minute"
                persian_timeframe_phrase = "۱ دقیقه‌ای"
            elif timeframe == "minute5":
                timeframe_description = "5-minute"
                persian_timeframe_phrase = "۵ دقیقه‌ای"
            elif timeframe == "minute15":
                timeframe_description = "15-minute"
                persian_timeframe_phrase = "۱۵ دقیقه‌ای"
            elif timeframe == "minute30":
                timeframe_description = "30-minute"
                persian_timeframe_phrase = "۳۰ دقیقه‌ای"
            elif timeframe == "hour1":
                timeframe_description = "1-hour"
                persian_timeframe_phrase = "۱ ساعته"
            elif timeframe == "hour4":
                timeframe_description = "4-hour"
                persian_timeframe_phrase = "۴ ساعته"
            elif timeframe == "hour8":
                timeframe_description = "8-hour"
                persian_timeframe_phrase = "۸ ساعته"
            elif timeframe == "hour12":
                timeframe_description = "12-hour"
                persian_timeframe_phrase = "۱۲ ساعته"
            elif timeframe == "day1":
                timeframe_description = "daily"
                persian_timeframe_phrase = "روزانه"
            elif timeframe == "week1":
                timeframe_description = "weekly"
                persian_timeframe_phrase = "هفتگی"
            elif timeframe == "month1":
                timeframe_description = "monthly"
                persian_timeframe_phrase = "ماهانه"
            else:
                timeframe_description = timeframe
                persian_timeframe_phrase = timeframe  # Fallback

        prompt = f"""
        You are a professional cryptocurrency technical analyst. Generate a detailed analysis for {pair} in Persian using the provided data.

        **Input Data:**
        Trading Pair: {pair}
        Timeframe: {persian_timeframe_phrase}
        
        ```
        {structured_data}
        ```

        **Output Requirements:**
        1. Language: **Persian (Farsi) only**
        2. Format: Telegram Markdown
        3. Follow this EXACT structure:

        📊 تحلیل کامل‌تر:

        ۱. خلاصه عمومی و وضعیت فعلی:
        - در آخرین کندل [timeframe] (تاریخ [date])، قیمت {pair} با [change_percentage] بسته شده است.
        - قیمت فعلی ([current_price]) در محدوده [position description relative to range] قرار دارد.
        - حجم معاملات در آخرین دوره [volume] بوده است.
        - نوسانات اخیر در سطح [volatility_percentage] قرار دارد.

        ۲. تحلیل تکنیکال جامع:
        - میانگین‌های متحرک (EMAs):
        - میانگین‌های متحرک کوتاه‌مدت (EMA_9 در [value] و EMA_21 در [value]) [trend_description] و [position_relative_to_price].
        - میانگین متحرک بلندمدت (EMA_50 در [value]) [trend_description] و [position_relative_to_price].
        - قیمت فعلی [position_description relative to EMAs].

        - اندیکاتورهای مومنتوم (RSI, MFI):
        - اندیکاتور RSI_14 با مقدار [value] در محدوده [overbought/oversold/neutral] قرار دارد و [trend_direction].
        - اندیکاتور MFI_14 با مقدار [value] در محدوده [description] قرار دارد و [trend_direction].

        - قدرت روند (ADX, DI+/DI-):
        - اندیکاتور ADX_14 با مقدار [value] نشان‌دهنده [strong/weak/ranging trend].
        - مقایسه DI+ ([value]) و DI- ([value]) نشان می‌دهد که [comparison and trend direction].

        - باندهای نوسان (Bollinger Bands):
        - قیمت فعلی ([price]) [position relative to bands] باندهای بولینگر قرار دارد.
        - باند بالایی در فاصله [percentage] بالای قیمت و باند پایینی در فاصله [percentage] پایین‌تر از قیمت قرار دارد.
        - پهنای باند [description of volatility].

        ۳. سطوح حمایت و مقاومت کلیدی:
        - بر اساس داده‌های ارائه شده، [resistance levels description].
        - سطوح حمایت مهم در پایین‌تر از قیمت فعلی شناسایی شده‌اند:
          - حمایت اول: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
          - حمایت دوم: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
        - قیمت فعلی [distance description from support/resistance levels].

        ۴. سناریوهای احتمالی و پیشنهاد معاملاتی:
        - با توجه به [indicator summary], سناریوی اصلی [bullish/bearish/neutral] است.
        - سناریوی صعودی: [bullish scenario description].
        - سناریوی نزولی: [bearish scenario description].
        - پیشنهاد معاملاتی عمومی: [trading recommendation based on analysis].

        ۵. ارزیابی ریسک:
        - ریسک اصلی در این تحلیل، [main risk factor].
        - شکست قاطع سطح [key level] می‌تواند [impact description].
        - انتشار اخبار مهم اقتصادی یا تغییرات ناگهانی در احساسات بازار کریپتو می‌تواند به سرعت تحلیل تکنیکال را تحت تاثیر قرار دهد.

        **Important Instructions:**
        - Use ONLY the provided structured data
        - Extract exact values from the data (EMAs, RSI, MFI, ADX, DI+, DI-, Bollinger Bands, Support/Resistance levels)
        - Provide detailed percentage calculations and comparisons
        - Use specific numbers and values throughout the analysis
        - Maintain professional Persian technical analysis terminology
        - Output ONLY the formatted analysis - no extra text
        """

        # Get the LLM client (defaults to OpenAI)
        llm_client = get_llm_client()

        # Generate the analysis using the LLM
        analysis = llm_client.generate_text(prompt)

        return analysis

    except Exception as e:
        logger.error(f"Error generating detailed analysis for {pair}: {str(e)}")
        raise
