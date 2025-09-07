"""
LLM Client module for interacting with Large Language Models.
This implementation uses OpenAI by default, but is designed to be easily swappable.
"""

import abc
import asyncio
import logging
from typing import Dict, Optional

# Import error types directly from openai package
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.config import AVALAI_API_BASE_URL, AVALAI_API_KEY, OPENAI_MODEL

# Configure logger
logger = logging.getLogger(__name__)


def unescape_markdownv2(text: str) -> str:
    """
    Remove existing MarkdownV2 escape sequences to prevent double escaping.
    Note: We don't unescape '*' and '_' as they should remain for formatting.

    Args:
        text: The text that may contain existing escape sequences

    Returns:
        Text with escape sequences removed
    """
    import re

    # Remove backslashes followed by special characters
    # This handles patterns like \\- or \\. or \\_
    # Note: We don't unescape '*' as it should remain for formatting
    special_chars = [
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
        "\\",
        "_",  # Include underscore in unescaping since we now escape it
    ]

    for char in special_chars:
        # Remove single backslash before the character (e.g., \\- -> -)
        if char == "\\":
            # Handle backslash specially - remove escaped backslashes
            text = text.replace("\\\\", "\\")
        else:
            pattern = r"\\" + re.escape(char)
            text = re.sub(pattern, char, text)

    return text


def escape_markdownv2(text: str) -> str:
    """
    Escape MarkdownV2 special characters for Telegram.
    First removes any existing escape sequences to prevent double escaping.
    Preserves formatting characters (*, _) and '>' characters at the beginning of lines for blockquotes.

    In MarkdownV2, these characters are special and must be escaped with backslash:
    []()~`>#+-=|{}.!

    Args:
        text: The text to escape

    Returns:
        Text with special characters escaped for MarkdownV2
    """
    # First remove any existing escape sequences
    text = unescape_markdownv2(text)

    # Characters that need escaping in MarkdownV2 (excluding '*' for bold formatting and '>' for blockquotes)
    # Note: We escape '_' because it's used for italic formatting in MarkdownV2, and we don't want
    # underscores in coin names like "sui_usdt" to be interpreted as italic formatting
    special_chars = [
        "\\",  # Backslash must be escaped first to avoid double escaping
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
        "_",  # Escape underscores to prevent italic formatting from coin names like "sui_usdt"
    ]

    # Escape all special characters except '*' (for bold) and '>' (for blockquotes)
    for char in special_chars:
        text = text.replace(char, f"\\{char}")

    # Handle '>' character specially - only escape if it's NOT at the beginning of a line
    # This preserves blockquote functionality while escaping '>' in other contexts
    # Split by lines and process each line individually
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        # If line starts with '>', don't escape the first '>'
        if line.lstrip().startswith(">"):
            # Find the first '>' after any leading whitespace
            stripped = line.lstrip()
            leading_space = line[: len(line) - len(stripped)]
            # Keep the first '>' unescaped, escape any others in the line
            rest_of_line = stripped[1:]  # Everything after the first '>'
            rest_escaped = rest_of_line.replace(">", "\\>")
            processed_line = leading_space + ">" + rest_escaped
        else:
            # Escape all '>' characters in non-blockquote lines
            processed_line = line.replace(">", "\\>")

        processed_lines.append(processed_line)

    text = "\n".join(processed_lines)

    return text


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients to enable easy provider switching."""

    @abc.abstractmethod
    async def generate_text(self, prompt: str) -> str:
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
            model: OpenAI model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
        """
        self.api_key = api_key
        self.model = model

        print(f"Initializing OpenAI client with model: {model}")

        if not api_key:
            logger.error("OpenAI API key not provided.")
            raise ValueError("OpenAI API key is required.")

        try:
            self.client = AsyncOpenAI(api_key=api_key, base_url=AVALAI_API_BASE_URL)
            logger.info(f"Initialized AsyncOpenAI client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {str(e)}")
            raise

    async def generate_text(
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

                # Make the async API call to OpenAI
                response = await self.client.chat.completions.create(
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
                        # Add debug logging for the full generated text
                        logger.debug("======== FULL GENERATED TEXT FROM LLM ========")
                        logger.debug(generated_text)
                        logger.debug("==============================================")
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
                    await asyncio.sleep(retry_delay)
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
                    await asyncio.sleep(retry_delay * 2)  # Wait longer for API errors
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
    provider: str = "openai", config: Optional[Dict] = None
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


async def generate_basic_analysis(
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
        3.  **IMPORTANT:** Do NOT escape any characters manually - this will be handled automatically.
        4.  **Structure:**
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
        analysis = await llm_client.generate_text(prompt)

        # Apply MarkdownV2 escaping to ensure Telegram compatibility
        escaped_analysis = escape_markdownv2(analysis)

        # Add debug logging for the escaped analysis
        logger.debug("======== ESCAPED ANALYSIS (BASIC) ========")
        logger.debug(escaped_analysis)
        logger.debug("==========================================")

        return escaped_analysis

    except Exception as e:
        logger.error(f"Error generating analysis for {pair}: {str(e)}")
        raise


async def generate_summarized_analysis(
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
        2. Format: MarkdownV2 for Telegram
           - MarkdownV2 formatting guide:
             * Bold text: Use **text** for bold formatting
             * Italic text: Use __text__ for italic formatting
             * Monospace/code: Use `text` for inline code
             * Strikethrough: Use ~~text~~ for strikethrough
             * Lists are NOT supported in Telegram MarkdownV2
             * Use bullet points with ▫️ or - symbols instead
           - IMPORTANT: Do NOT escape any characters manually - this will be handled automatically
        3. Keep it SHORT - maximum 10-12 lines
        4. Follow this EXACT structure:

        📊 تحلیل {pair} - تایم‌فریم {persian_timeframe_phrase}

        ▫️وضعیت کلی:
        - قیمت لحظه‌ای: [current_price from "Current Market Price (Live)" section if available, otherwise use latest Close price]
        - روند بلندمدت --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - روند کوتاه‌مدت --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - حمایت مهم بعدی: [support_level]
        - مقاومت مهم بعدی: [resistance_level]

        💭 توصیه عملی:
        - [Short practical recommendation based on indicators]
        - نقطه ورود احتمالی: [specific price range based on EMA levels, support levels, or pullback zones - e.g., "محدوده 105000-105500" or "در صورت پولبک به محدوده 105235-105500"]
        - سطح کلیدی برای تایید: [confirmation level]
        - سطح ریسک: [متوسط/بالا/پایین]

        ⚠️ نکات مهم:
        - [Key warning or note about EMA levels/resistance failure]
        - [Important level that could change the outlook]

        **Important Instructions:**
        - Use ONLY the provided structured data
        - For trend strength, use exactly one of: قوی, متوسط, ضعیف
        - For trend direction, use exactly one of: صعودی, نزولی, خنثی
        - Extract current price: Look for "Current Market Price (Live)" section first. If it exists, use the "Current Price" value. If not, use the Close price from "Latest OHLCV Data" section
        - Use identified Support/Resistance levels from the data
        - Base recommendations on ALL available indicators (EMAs, RSI, MFI, ADX, DI+/DI-, Bollinger Bands) but keep mentions brief since this is a summarized report
        - You may briefly reference other indicators (RSI overbought/oversold, Bollinger Band position, ADX trend strength) when relevant for trend assessment or entry suggestions, but focus primarily on EMAs and Support/Resistance levels
        - For entry points: ALWAYS provide specific price ranges. Use EMA levels, support levels, or create reasonable pullback zones. Examples: "محدوده 105000-105500", "در صورت پولبک به محدوده EMA_21 (105235-105500)", "نزدیک سطح حمایت 104800-105200"
        - Keep recommendations practical and specific
        - Output ONLY the formatted analysis - no extra text
        """

        # Get the LLM client
        llm_client = get_llm_client()

        # Generate the analysis using the LLM
        analysis = await llm_client.generate_text(prompt)

        # Apply MarkdownV2 escaping to ensure Telegram compatibility
        escaped_analysis = escape_markdownv2(analysis)

        # Add "نتیجه" header to the final text
        final_analysis = "**نتیجه**\n\n" + escaped_analysis

        # Add debug logging for the escaped analysis
        logger.debug("======== ESCAPED ANALYSIS (SUMMARIZED) ========")
        logger.debug(final_analysis)
        logger.debug("===============================================")

        return final_analysis

    except Exception as e:
        logger.error(f"Error generating summarized analysis for {pair}: {str(e)}")
        raise


async def generate_combined_analysis(
    pair: str,
    structured_data: str,
    timeframe: Optional[str] = None,
    use_placeholders: bool = False,
) -> Dict[str, str]:
    """
    Generate both detailed and summarized analysis in a single LLM call.

    Args:
        pair: The trading pair (e.g., 'BTCUSDT')
        structured_data: A string containing formatted OHLCV data,
                         technical indicators, and S/R levels.
        timeframe: The timeframe of the data (e.g., 'day1', 'hour4')
        use_placeholders: If True, instruct LLM to use placeholders for current price

    Returns:
        Dictionary containing both analyses:
        {
            "detailed_analysis": "The detailed analysis text in Persian",
            "summarized_analysis": "The summarized analysis text in Persian"
        }

    Raises:
        Exception: If there's an error in the analysis generation.
    """
    try:
        # Get Persian timeframe phrases for both analyses
        persian_timeframe_phrase_detailed = ""
        persian_timeframe_phrase_summarized = ""

        if timeframe:
            if timeframe == "minute1":
                persian_timeframe_phrase_detailed = "۱ دقیقه‌ای"
                persian_timeframe_phrase_summarized = "1M"
            elif timeframe == "minute5":
                persian_timeframe_phrase_detailed = "۵ دقیقه‌ای"
                persian_timeframe_phrase_summarized = "5M"
            elif timeframe == "minute15":
                persian_timeframe_phrase_detailed = "۱۵ دقیقه‌ای"
                persian_timeframe_phrase_summarized = "15M"
            elif timeframe == "minute30":
                persian_timeframe_phrase_detailed = "۳۰ دقیقه‌ای"
                persian_timeframe_phrase_summarized = "30M"
            elif timeframe == "hour1":
                persian_timeframe_phrase_detailed = "۱ ساعته"
                persian_timeframe_phrase_summarized = "1H"
            elif timeframe == "hour4":
                persian_timeframe_phrase_detailed = "۴ ساعته"
                persian_timeframe_phrase_summarized = "4H"
            elif timeframe == "hour8":
                persian_timeframe_phrase_detailed = "۸ ساعته"
                persian_timeframe_phrase_summarized = "8H"
            elif timeframe == "hour12":
                persian_timeframe_phrase_detailed = "۱۲ ساعته"
                persian_timeframe_phrase_summarized = "12H"
            elif timeframe == "day1":
                persian_timeframe_phrase_detailed = "روزانه"
                persian_timeframe_phrase_summarized = "1D"
            elif timeframe == "week1":
                persian_timeframe_phrase_detailed = "هفتگی"
                persian_timeframe_phrase_summarized = "1W"
            elif timeframe == "month1":
                persian_timeframe_phrase_detailed = "ماهانه"
                persian_timeframe_phrase_summarized = "1M"
            else:
                persian_timeframe_phrase_detailed = timeframe
                persian_timeframe_phrase_summarized = timeframe

        prompt = f"""
        You are a professional cryptocurrency technical analyst. Generate BOTH a detailed analysis and a summarized analysis for {pair} in Persian using the provided data.

        **FORMATTING EXAMPLE**: 
        Instead of: "قیمت لحظه‌ای: 4303.0900"
        Write: "**قیمت لحظه‌ای:** 4303.0900"
        
        Instead of: "روند بلندمدت --> متوسط صعودی"
        Write: "**روند بلندمدت** --> متوسط صعودی"

        **Input Data:**
        Trading Pair: {pair}
        Timeframe: {persian_timeframe_phrase_detailed}
        
        ```
        {structured_data}
        ```

        {"**IMPORTANT - Price Placeholder Instructions:**" if use_placeholders else ""}
        {"""For current price references, use exactly CURRENTPRICE instead of actual price values.
        This placeholder will be replaced with the actual current price after generation.
        
        Examples:AnalyzeResponse
        - قیمت لحظه‌ای: CURRENTPRICE
        - قیمت فعلی (CURRENTPRICE) در محدوده...
        - Instead of writing "قیمت فعلی (105000)" write "قیمت فعلی (CURRENTPRICE)"
        
        CRITICAL: Always use CURRENTPRICE for any current price reference.""" if use_placeholders else ""}

        **Output Requirements:**
        1. Language: **Persian (Farsi) only**
        2. Format: MarkdownV2 for Telegram
           - MarkdownV2 formatting guide:
             * Bold text: Use **text** for bold formatting
             * Italic text: Use __text__ for italic formatting
             * Monospace/code: Use `text` for inline code
             * Strikethrough: Use ~~text~~ for strikethrough
             * Quote text: Use >text at the beginning of a line for quotes
             * Lists are NOT supported in Telegram MarkdownV2
             * Use bullet points with ▫️ or - symbols instead
           - IMPORTANT: Do NOT escape any characters manually - this will be handled automatically
           - **CRITICAL**: You MUST use **bold** formatting for all key terms and labels in your response
        3. **CRITICAL**: You MUST return a valid JSON object with exactly this structure:
        {{
            "detailed_analysis": "DETAILED_ANALYSIS_CONTENT_HERE",
            "summarized_analysis": "SUMMARIZED_ANALYSIS_CONTENT_HERE"
        }}

        **Detailed Analysis Structure:**
        Follow this EXACT structure for the detailed_analysis field:

        📊 تحلیل کامل‌تر:

        >۱. خلاصه عمومی و وضعیت فعلی:
        - در آخرین کندل [{persian_timeframe_phrase_detailed}] (تاریخ [date])، قیمت {pair} با [change_percentage] بسته شده است.
        - **قیمت فعلی** ([current_price]) در محدوده [position description relative to range] قرار دارد.
        - **حجم معاملات** در آخرین دوره [volume] بوده است.
        - **نوسانات اخیر** در سطح [volatility_percentage] قرار دارد.

        >۲. تحلیل تکنیکال جامع:
        - **میانگین‌های متحرک (EMAs):**
        - میانگین‌های متحرک کوتاه‌مدت (EMA_9 در [value] و EMA_21 در [value]) [trend_description] و [position_relative_to_price].
        - میانگین متحرک بلندمدت (EMA_50 در [value]) [trend_description] و [position_relative_to_price].
        - **قیمت فعلی** [position_description relative to EMAs].

        - **اندیکاتورهای مومنتوم (RSI, MFI):**
        - اندیکاتور RSI_14 با مقدار [value] در محدوده [overbought/oversold/neutral] قرار دارد و [trend_direction].
        - اندیکاتور MFI_14 با مقدار [value] در محدوده [description] قرار دارد و [trend_direction].

        - **قدرت روند (ADX, DI+/DI-):**
        - اندیکاتور ADX_14 با مقدار [value] نشان‌دهنده [strong/weak/ranging trend].
        - مقایسه DI+ ([value]) و DI- ([value]) نشان می‌دهد که [comparison and trend direction].

        - **باندهای نوسان (Bollinger Bands):**
        - **قیمت فعلی** ([price]) [position relative to bands] باندهای بولینگر قرار دارد.
        - باند بالایی در فاصله [percentage] بالای قیمت و باند پایینی در فاصله [percentage] پایین‌تر از قیمت قرار دارد.
        - **پهنای باند** [description of volatility].

        >۳. سطوح حمایت و مقاومت کلیدی:
        - بر اساس داده‌های ارائه شده، [resistance levels description].
        - سطوح حمایت مهم در پایین‌تر از قیمت فعلی شناسایی شده‌اند:
          - حمایت اول: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
          - حمایت دوم: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
        - قیمت فعلی [distance description from support/resistance levels].

        >۴. سناریوهای احتمالی و پیشنهاد معاملاتی:
        - با توجه به [indicator summary], سناریوی اصلی [bullish/bearish/neutral] است.
        - سناریوی صعودی: [bullish scenario description].
        - سناریوی نزولی: [bearish scenario description].
        - پیشنهاد معاملاتی عمومی: [trading recommendation based on analysis].

        >۵. ارزیابی ریسک:
        - ریسک اصلی در این تحلیل، [main risk factor].
        - شکست قاطع سطح [key level] می‌تواند [impact description].
        - انتشار اخبار مهم اقتصادی یا تغییرات ناگهانی در احساسات بازار کریپتو می‌تواند به سرعت تحلیل تکنیکال را تحت تاثیر قرار دهد.

        **Summarized Analysis Structure:**
        Follow this EXACT structure for the summarized_analysis field:

        📊 تحلیل {pair} - تایم‌فریم {persian_timeframe_phrase_summarized}

        ▫️وضعیت کلی:
        - **قیمت لحظه‌ای:** [current_price from "Current Market Price (Live)" section if available, otherwise use latest Close price]
        - **روند بلندمدت** --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - **روند کوتاه‌مدت** --> [قوی/متوسط/ضعیف] [صعودی/نزولی/خنثی]
        - **حمایت مهم بعدی:** [support_level]
        - **مقاومت مهم بعدی:** [resistance_level]

        💭 توصیه عملی:
        - [Short practical recommendation based on indicators]
        - **نقطه ورود احتمالی:** [specific price range based on EMA levels, support levels, or pullback zones - e.g., "محدوده 105000-105500" or "در صورت پولبک به محدوده 105235-105500"]
        - **سطح کلیدی برای تایید:** [confirmation level]
        - **سطح ریسک:** [متوسط/بالا/پایین]

        ⚠️ نکات مهم:
        - [Key warning or note about EMA levels/resistance failure]
        - [Important level that could change the outlook]

        **Important Instructions:**
        - Use ONLY the provided structured data
        - Extract exact values from the data (EMAs, RSI, MFI, ADX, DI+, DI-, Bollinger Bands, Support/Resistance levels)
        - For trend strength, use exactly one of: قوی, متوسط, ضعیف
        - For trend direction, use exactly one of: صعودی, نزولی, خنثی
        - Extract current price: Look for "Current Market Price (Live)" section first. If it exists, use the "Current Price" value. If not, use the Close price from "Latest OHLCV Data" section
        - Use identified Support/Resistance levels from the data
        - Base recommendations on ALL available indicators (EMAs, RSI, MFI, ADX, DI+/DI-, Bollinger Bands) but keep mentions brief in summarized analysis
        - For entry points in summarized analysis: ALWAYS provide specific price ranges. Use EMA levels, support levels, or create reasonable pullback zones. Examples: "محدوده 105000-105500", "در صورت پولبک به محدوده EMA_21 (105235-105500)", "نزدیک سطح حمایت 104800-105200"
        - Keep summarized recommendations practical and specific
        - Provide detailed percentage calculations and comparisons in detailed analysis
        - Use specific numbers and values throughout both analyses
        - Maintain professional Persian technical analysis terminology
        - **CRITICAL FORMATTING**: You MUST use **bold** formatting for key terms. Examples:
          * "قیمت لحظه‌ای" should be "**قیمت لحظه‌ای**"
          * "روند بلندمدت" should be "**روند بلندمدت**"
          * "روند کوتاه‌مدت" should be "**روند کوتاه‌مدت**"
          * "حمایت مهم بعدی" should be "**حمایت مهم بعدی**"
          * "مقاومت مهم بعدی" should be "**مقاومت مهم بعدی**"
          * "نقطه ورود احتمالی" should be "**نقطه ورود احتمالی**"
          * "سطح کلیدی برای تایید" should be "**سطح کلیدی برای تایید**"
          * "سطح ریسک" should be "**سطح ریسک**"
          * "قیمت فعلی" should be "**قیمت فعلی**"
          * "حجم معاملات" should be "**حجم معاملات**"
          * "نوسانات اخیر" should be "**نوسانات اخیر**"
          * "میانگین‌های متحرک (EMAs)" should be "**میانگین‌های متحرک (EMAs)**"
          * "اندیکاتورهای مومنتوم (RSI, MFI)" should be "**اندیکاتورهای مومنتوم (RSI, MFI)**"
          * "قدرت روند (ADX, DI+/DI-)" should be "**قدرت روند (ADX, DI+/DI-)**"
          * "باندهای نوسان (Bollinger Bands)" should be "**باندهای نوسان (Bollinger Bands)**"
          * "پهنای باند" should be "**پهنای باند**"
        - **CRITICAL**: Return ONLY the JSON object - no extra text before or after
        """

        # Get the LLM client
        llm_client = get_llm_client()

        # Generate the combined analysis using the LLM
        response_text = await llm_client.generate_text(prompt)

        # Parse the JSON response
        import json

        try:
            # Clean the response text - remove any potential markdown code blocks
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parse JSON
            response_json = json.loads(cleaned_response)

            # Validate the response structure
            if not isinstance(response_json, dict):
                raise ValueError("Response is not a dictionary")
            if "detailed_analysis" not in response_json:
                raise ValueError("Missing 'detailed_analysis' field in response")
            if "summarized_analysis" not in response_json:
                raise ValueError("Missing 'summarized_analysis' field in response")

            # Extract and process both analyses
            detailed_analysis = response_json["detailed_analysis"].strip()
            summarized_analysis = response_json["summarized_analysis"].strip()

            # Apply MarkdownV2 escaping to both analyses
            escaped_detailed = escape_markdownv2(detailed_analysis)
            escaped_summarized = escape_markdownv2(summarized_analysis)

            # Add "نتیجه" header to the summarized analysis
            final_summarized = "**نتیجه**\n\n" + escaped_summarized

            # Add debug logging for the escaped analyses
            logger.debug("======== ESCAPED ANALYSIS (COMBINED - DETAILED) ========")
            logger.debug(escaped_detailed)
            logger.debug("======================================================")
            logger.debug("======== ESCAPED ANALYSIS (COMBINED - SUMMARIZED) ========")
            logger.debug(final_summarized)
            logger.debug("========================================================")

            return {
                "detailed_analysis": escaped_detailed,
                "summarized_analysis": final_summarized,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error("Raw response: %s", response_text[:500])
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
        except KeyError as e:
            logger.error(f"Missing required field in JSON response: {str(e)}")
            if "response_json" in locals():
                logger.error("Response structure keys: %s", list(response_json.keys()))
            else:
                logger.error("Response structure: N/A")
            raise ValueError(f"Invalid response structure: missing {str(e)}")

    except Exception as e:
        logger.error(f"Error generating combined analysis for {pair}: {str(e)}")
        raise


async def generate_detailed_analysis(
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
        # Get Persian timeframe phrase
        persian_timeframe_phrase = ""
        if timeframe:
            if timeframe == "minute1":
                persian_timeframe_phrase = "۱ دقیقه‌ای"
            elif timeframe == "minute5":
                persian_timeframe_phrase = "۵ دقیقه‌ای"
            elif timeframe == "minute15":
                persian_timeframe_phrase = "۱۵ دقیقه‌ای"
            elif timeframe == "minute30":
                persian_timeframe_phrase = "۳۰ دقیقه‌ای"
            elif timeframe == "hour1":
                persian_timeframe_phrase = "۱ ساعته"
            elif timeframe == "hour4":
                persian_timeframe_phrase = "۴ ساعته"
            elif timeframe == "hour8":
                persian_timeframe_phrase = "۸ ساعته"
            elif timeframe == "hour12":
                persian_timeframe_phrase = "۱۲ ساعته"
            elif timeframe == "day1":
                persian_timeframe_phrase = "روزانه"
            elif timeframe == "week1":
                persian_timeframe_phrase = "هفتگی"
            elif timeframe == "month1":
                persian_timeframe_phrase = "ماهانه"
            else:
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
        2. Format: MarkdownV2 for Telegram
           - MarkdownV2 formatting guide:
             * Bold text: Use **text** for bold formatting
             * Italic text: Use __text__ for italic formatting
             * Monospace/code: Use `text` for inline code
             * Strikethrough: Use ~~text~~ for strikethrough
             * Quote text: Use >text at the beginning of a line for quotes
             * Lists are NOT supported in Telegram MarkdownV2
             * Use bullet points with ▫️ or - symbols instead
           - IMPORTANT: Do NOT escape any characters manually - this will be handled automatically
        3. Follow this EXACT structure:

        📊 تحلیل کامل‌تر:

        >۱. خلاصه عمومی و وضعیت فعلی:
        - در آخرین کندل [timeframe] (تاریخ [date])، قیمت {pair} با [change_percentage] بسته شده است.
        - قیمت فعلی ([current_price]) در محدوده [position description relative to range] قرار دارد.
        - حجم معاملات در آخرین دوره [volume] بوده است.
        - نوسانات اخیر در سطح [volatility_percentage] قرار دارد.

        >۲. تحلیل تکنیکال جامع:
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

        >۳. سطوح حمایت و مقاومت کلیدی:
        - بر اساس داده‌های ارائه شده، [resistance levels description].
        - سطوح حمایت مهم در پایین‌تر از قیمت فعلی شناسایی شده‌اند:
          - حمایت اول: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
          - حمایت دوم: در حدود قیمت [level] ([percentage] پایین‌تر از قیمت فعلی).
        - قیمت فعلی [distance description from support/resistance levels].

        >۴. سناریوهای احتمالی و پیشنهاد معاملاتی:
        - با توجه به [indicator summary], سناریوی اصلی [bullish/bearish/neutral] است.
        - سناریوی صعودی: [bullish scenario description].
        - سناریوی نزولی: [bearish scenario description].
        - پیشنهاد معاملاتی عمومی: [trading recommendation based on analysis].

        >۵. ارزیابی ریسک:
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
        analysis = await llm_client.generate_text(prompt)

        # Apply MarkdownV2 escaping to ensure Telegram compatibility
        escaped_analysis = escape_markdownv2(analysis)

        # Add debug logging for the escaped analysis
        logger.debug("======== ESCAPED ANALYSIS (DETAILED) ========")
        logger.debug(escaped_analysis)
        logger.debug("============================================")

        return escaped_analysis

    except Exception as e:
        logger.error(f"Error generating detailed analysis for {pair}: {str(e)}")
        raise
