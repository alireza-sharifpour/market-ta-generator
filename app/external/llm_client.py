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

from app.config import OPENAI_API_KEY, OPENAI_MODEL

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

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_MODEL):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (e.g., 'gpt-3.5-turbo', 'gpt-4o')
        """
        self.api_key = api_key
        self.model = model

        if not api_key:
            logger.error("OpenAI API key not provided.")
            raise ValueError("OpenAI API key is required.")

        try:
            self.client = OpenAI(api_key=api_key)
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
                logger.debug(f"Sending prompt to OpenAI: {prompt[:100]}...")

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


def generate_basic_analysis(pair: str, data_summary: str) -> str:
    """
    Generate a basic analysis of the cryptocurrency pair data.

    Args:
        pair: The trading pair (e.g., 'BTCUSDT')
        data_summary: Processed data summary from the data_processor

    Returns:
        The generated analysis text

    Raises:
        Exception: If there's an error in the analysis generation
    """
    try:
        # Construct the prompt for the LLM
        prompt = f"""
        Analyze the following daily OHLCV (Open, High, Low, Close, Volume) data for the {pair} trading pair and provide a brief summary:

        {data_summary}

        Please provide:
        1. A general market summary based on the data
        2. Key price levels and trends
        3. An overall sentiment (bullish, bearish, or neutral)
        """

        # Get the LLM client (defaults to OpenAI)
        llm_client = get_llm_client()

        # Generate the analysis using the LLM
        analysis = llm_client.generate_text(prompt)

        return analysis

    except Exception as e:
        logger.error(f"Error generating analysis for {pair}: {str(e)}")
        raise
