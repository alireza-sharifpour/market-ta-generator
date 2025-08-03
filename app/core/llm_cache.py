"""
LLM Cache module providing intelligent caching for LLM analysis with placeholder support.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from app.config import CACHE_KEY_PREFIX, CACHE_PLACEHOLDERS, CACHE_TTL_SETTINGS
from app.core.cache_service import get_cache_service
from app.core.data_processor import prepare_llm_input_for_cache
from app.external.llm_client import generate_combined_analysis, escape_markdownv2

# Set up logging
logger = logging.getLogger(__name__)


class LLMCacheError(Exception):
    """Exception raised for LLM cache-related errors."""
    pass


class LLMCache:
    """
    Specialized cache manager for LLM analysis with placeholder support.
    """
    
    def __init__(self):
        """Initialize LLM cache manager."""
        self.cache_service = None
        
    async def _get_cache_service(self):
        """Get or initialize cache service."""
        if self.cache_service is None:
            self.cache_service = await get_cache_service()
            if self.cache_service.redis_client is None:
                await self.cache_service.initialize()
        return self.cache_service
    
    def generate_data_hash(
        self, 
        df_with_indicators: DataFrame, 
        sr_levels: Dict[str, List[float]], 
        timeframe: str
    ) -> str:
        """
        Generate a stable hash from market data that affects analysis.
        
        Args:
            df_with_indicators: DataFrame with OHLCV data and technical indicators
            sr_levels: Dictionary with support and resistance levels
            timeframe: Trading timeframe
            
        Returns:
            Hash string representing the market data state
        """
        try:
            # Get the last 5 candles for cache key (enough to capture recent changes)
            recent_candles = df_with_indicators.tail(5)
            
            # Extract key data points that affect analysis
            cache_data = {
                # Recent OHLCV data (exclude exact timestamps, use relative positions)
                "recent_ohlcv": [
                    {
                        "open": round(float(row["Open"]), 6),
                        "high": round(float(row["High"]), 6),
                        "low": round(float(row["Low"]), 6),
                        "close": round(float(row["Close"]), 6),
                        "volume": round(float(row["Volume"]), 2),
                    }
                    for _, row in recent_candles.iterrows()
                ],
                
                # Current technical indicator values (rounded to reduce fragmentation)
                "indicators": {
                    "ema_9": round(float(recent_candles.iloc[-1]["EMA_9"]), 6),
                    "ema_21": round(float(recent_candles.iloc[-1]["EMA_21"]), 6),
                    "ema_50": round(float(recent_candles.iloc[-1]["EMA_50"]), 6),
                    "rsi": round(float(recent_candles.iloc[-1]["RSI_14"]), 2),
                    "adx": round(float(recent_candles.iloc[-1]["ADX_14"]), 2),
                    "mfi": round(float(recent_candles.iloc[-1]["MFI_14"]), 2),
                    "bb_upper": round(float(recent_candles.iloc[-1]["BBU_20_2.0"]), 6),
                    "bb_middle": round(float(recent_candles.iloc[-1]["BBM_20_2.0"]), 6),
                    "bb_lower": round(float(recent_candles.iloc[-1]["BBL_20_2.0"]), 6),
                    "di_plus": round(float(recent_candles.iloc[-1]["DMP_14"]), 2),
                    "di_minus": round(float(recent_candles.iloc[-1]["DMN_14"]), 2),
                },
                
                # Support/Resistance levels (rounded to reduce fragmentation)
                "sr_levels": {
                    "support": [round(level, 4) for level in sr_levels.get("support", [])],
                    "resistance": [round(level, 4) for level in sr_levels.get("resistance", [])],
                },
                
                # Timeframe for context
                "timeframe": timeframe,
                
                # Time bucket (15-minute buckets for high freq, hourly for low freq)
                "time_bucket": self._get_time_bucket(timeframe),
            }
            
            # Create deterministic hash
            data_str = json.dumps(cache_data, sort_keys=True)
            hash_obj = hashlib.sha256(data_str.encode('utf-8'))
            data_hash = hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
            
            logger.debug(f"Generated data hash: {data_hash} for timeframe: {timeframe}")
            return data_hash
            
        except Exception as e:
            logger.error(f"Error generating data hash: {e}")
            # Return a unique hash to avoid cache conflicts
            return hashlib.sha256(str(e).encode()).hexdigest()[:16]
    
    def _get_time_bucket(self, timeframe: str) -> str:
        """
        Get time bucket for cache key to group requests within time windows.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            Time bucket string
        """
        import time
        current_time = int(time.time())
        
        # Define bucket sizes (in seconds) for different timeframes
        bucket_sizes = {
            "minute1": 300,    # 5-minute buckets
            "minute5": 600,    # 10-minute buckets
            "minute15": 900,   # 15-minute buckets
            "minute30": 1800,  # 30-minute buckets
            "hour1": 3600,     # 1-hour buckets
            "hour4": 7200,     # 2-hour buckets
            "hour8": 14400,    # 4-hour buckets
            "hour12": 21600,   # 6-hour buckets
            "day1": 43200,     # 12-hour buckets
            "week1": 86400,    # 24-hour buckets
            "month1": 172800,  # 48-hour buckets
        }
        
        bucket_size = bucket_sizes.get(timeframe, 3600)  # Default 1-hour bucket
        bucket = current_time // bucket_size
        
        return f"{timeframe}_{bucket}"
    
    def generate_cache_key(
        self, 
        pair: str, 
        df_with_indicators: DataFrame, 
        sr_levels: Dict[str, List[float]], 
        timeframe: str
    ) -> str:
        """
        Generate consistent cache key from market data.
        
        Args:
            pair: Trading pair
            df_with_indicators: DataFrame with indicators
            sr_levels: Support/resistance levels
            timeframe: Trading timeframe
            
        Returns:
            Cache key string
        """
        try:
            data_hash = self.generate_data_hash(df_with_indicators, sr_levels, timeframe)
            cache_key = f"{CACHE_KEY_PREFIX}:llm_analysis:{pair.lower()}:{timeframe}:{data_hash}"
            
            logger.debug(f"Generated cache key: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            # Fallback to a basic key to avoid errors
            import time
            fallback_key = f"{CACHE_KEY_PREFIX}:llm_analysis:{pair.lower()}:{timeframe}:fallback_{int(time.time())}"
            return fallback_key
    
    def replace_price_placeholders(
        self, 
        analysis: Dict[str, str], 
        current_price: Optional[float]
    ) -> Dict[str, str]:
        """
        Replace price placeholders with actual current price.
        
        Args:
            analysis: Analysis dictionary with placeholder content
            current_price: Current price to inject
            
        Returns:
            Analysis dictionary with placeholders replaced
        """
        try:
            if current_price is None:
                logger.warning("No current price provided for placeholder replacement")
                current_price_str = "N/A"
            else:
                # Format price based on magnitude (same as data_processor.py)
                if current_price >= 1:
                    current_price_str = f"{current_price:.4f}"
                elif current_price >= 0.01:
                    current_price_str = f"{current_price:.6f}"
                elif current_price >= 0.0001:
                    current_price_str = f"{current_price:.8f}"
                else:
                    current_price_str = f"{current_price:.2e}"
                
                # Apply MarkdownV2 escaping to the price string
                escaped_price_str = escape_markdownv2(current_price_str)
            
            # Replace placeholders in both detailed and summarized analysis
            result = {}
            for key, content in analysis.items():
                if isinstance(content, str):
                    # Replace current price placeholder with escaped price
                    updated_content = content.replace(
                        CACHE_PLACEHOLDERS["current_price"], 
                        escaped_price_str
                    )
                    
                    # Note: We can add more placeholder replacements here in the future
                    # For example: price_change_24h, volume_24h
                    
                    result[key] = updated_content
                else:
                    result[key] = content
            
            logger.debug(f"Replaced placeholders with current price: {current_price_str} (escaped: {escaped_price_str})")
            return result
            
        except Exception as e:
            logger.error(f"Error replacing placeholders: {e}")
            return analysis  # Return original on error
    
    def get_cache_ttl(self, timeframe: str) -> int:
        """
        Get cache TTL based on timeframe.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            TTL in seconds
        """
        return CACHE_TTL_SETTINGS.get(timeframe, CACHE_TTL_SETTINGS["default"])
    
    async def get_or_generate(
        self,
        pair: str,
        df_with_indicators: DataFrame,
        sr_levels: Dict[str, List[float]],
        current_price: Optional[float],
        timeframe: str
    ) -> Dict[str, str]:
        """
        Get analysis from cache or generate new one with LLM.
        
        Args:
            pair: Trading pair
            df_with_indicators: DataFrame with indicators
            sr_levels: Support/resistance levels
            current_price: Current price for placeholder replacement
            timeframe: Trading timeframe
            
        Returns:
            Dictionary with detailed_analysis and summarized_analysis
        """
        try:
            # Generate cache key
            cache_key = self.generate_cache_key(pair, df_with_indicators, sr_levels, timeframe)
            
            # Try to get from cache
            cache_service = await self._get_cache_service()
            cached_analysis = await cache_service.get(cache_key)
            
            if cached_analysis:
                logger.info(f"Cache HIT for {pair} ({timeframe})")
                
                # Replace placeholders with current price
                analysis = self.replace_price_placeholders(cached_analysis, current_price)
                
                return analysis
            
            # Cache miss - generate new analysis
            logger.info(f"Cache MISS for {pair} ({timeframe}) - generating new analysis")
            
            # Prepare LLM input WITHOUT current price for cache consistency
            structured_llm_input = prepare_llm_input_for_cache(df_with_indicators, sr_levels)
            
            # Generate analysis with LLM
            analysis = await generate_combined_analysis(
                pair, structured_llm_input, timeframe=timeframe, use_placeholders=True
            )
            
            # Cache the analysis with placeholders
            ttl = self.get_cache_ttl(timeframe)
            await cache_service.set(cache_key, analysis, ttl)
            
            logger.info(f"Cached new analysis for {pair} ({timeframe}) with TTL {ttl}s")
            
            # Replace placeholders with current price before returning
            final_analysis = self.replace_price_placeholders(analysis, current_price)
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Error in get_or_generate for {pair} ({timeframe}): {e}")
            # Fallback to direct LLM generation without cache
            return await self._generate_without_cache(
                pair, df_with_indicators, sr_levels, current_price, timeframe
            )
    
    async def _generate_without_cache(
        self,
        pair: str,
        df_with_indicators: DataFrame,
        sr_levels: Dict[str, List[float]],
        current_price: Optional[float],
        timeframe: str
    ) -> Dict[str, str]:
        """
        Generate analysis without caching (fallback method).
        
        Args:
            pair: Trading pair
            df_with_indicators: DataFrame with indicators
            sr_levels: Support/resistance levels
            current_price: Current price
            timeframe: Trading timeframe
            
        Returns:
            Dictionary with detailed_analysis and summarized_analysis
        """
        try:
            logger.warning(f"Generating analysis without cache for {pair} ({timeframe})")
            
            # Prepare complete LLM input WITH current price (traditional approach)
            from app.core.data_processor import prepare_llm_input_phase2
            
            # Format current price data if available
            current_price_data = None
            if current_price is not None:
                current_price_data = {
                    "ticker": {
                        "latest": current_price
                    }
                }
            
            structured_llm_input = prepare_llm_input_phase2(
                df_with_indicators, sr_levels, current_price_data
            )
            
            # Generate analysis with LLM (without placeholders)
            analysis = await generate_combined_analysis(
                pair, structured_llm_input, timeframe=timeframe, use_placeholders=False
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fallback generation for {pair} ({timeframe}): {e}")
            # Return error response
            return {
                "detailed_analysis": f"خطا در تولید تحلیل: {str(e)}",
                "summarized_analysis": f"خطا در تولید تحلیل: {str(e)}"
            }
    
    async def invalidate_cache(self, pair: str, timeframe: str) -> bool:
        """
        Invalidate cache for a specific pair and timeframe.
        
        Args:
            pair: Trading pair
            timeframe: Trading timeframe
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: Since we don't know the exact data hash, we can't delete specific keys
            # This would require implementing a pattern-based deletion with the cache service
            # For now, we'll log this operation
            logger.info(f"Cache invalidation requested for {pair} ({timeframe})")
            
            # Note: Redis pattern deletion could be implemented here if needed
            # pattern = f"{CACHE_KEY_PREFIX}:llm_analysis:{pair.lower()}:{timeframe}:*"
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {pair} ({timeframe}): {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_service = await self._get_cache_service()
            return await cache_service.get_stats()
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Global LLM cache instance
llm_cache = LLMCache()