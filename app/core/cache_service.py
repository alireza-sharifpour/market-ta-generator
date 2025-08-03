"""
Cache Service module providing async Redis operations with connection pooling and error handling.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from app.config import CACHE_ENABLED, REDIS_URL

# Set up logging
logger = logging.getLogger(__name__)

# Check Redis availability and import types
try:
    import redis.asyncio
    from redis.asyncio import ConnectionPool, Redis

    REDIS_AVAILABLE = True
except ImportError:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from redis.asyncio import ConnectionPool, Redis
    else:
        Redis = Any  # type: ignore
        ConnectionPool = Any  # type: ignore
    redis = None  # type: ignore
    REDIS_AVAILABLE = False


class CacheError(Exception):
    """Exception raised for cache-related errors."""

    pass


class CacheService:
    """
    Async Redis cache service with connection pooling and error handling.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        """
        Initialize cache service with Redis connection.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self._enabled = CACHE_ENABLED and REDIS_AVAILABLE

    async def initialize(self) -> None:
        """
        Initialize Redis connection pool.
        """
        if not self._enabled:
            if not REDIS_AVAILABLE:
                logger.info("Redis packages not available, cache disabled")
            else:
                logger.info("Cache is disabled, skipping Redis initialization")
            return

        if not REDIS_AVAILABLE:
            logger.error("Redis packages not available")
            return

        try:
            # Import redis.asyncio module
            import redis.asyncio as redis_asyncio

            # Create connection pool
            self.connection_pool = redis_asyncio.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=10,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            # Create Redis client
            self.redis_client = redis_asyncio.Redis(
                connection_pool=self.connection_pool
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis")

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self.redis_client = None
            self.connection_pool = None
            # Don't raise here - let the application continue without cache

    async def close(self) -> None:
        """
        Close Redis connection and cleanup resources.
        """
        if self.redis_client:
            try:
                await self.redis_client.aclose()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

        if self.connection_pool:
            try:
                await self.connection_pool.aclose()
                logger.info("Redis connection pool closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection pool: {e}")

    def _is_available(self) -> bool:
        """
        Check if cache is enabled and Redis client is available.

        Returns:
            True if cache is available, False otherwise
        """
        return self._enabled and self.redis_client is not None

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value as dictionary or None if not found or error
        """
        if not self._is_available():
            return None

        try:
            # Add timeout to prevent hanging
            if self.redis_client:
                value = await asyncio.wait_for(self.redis_client.get(key), timeout=5.0)

                if value is None:
                    return None

                # Parse JSON
                result = json.loads(value)
                logger.debug(f"Cache hit for key: {key}")
                return result
            return None

        except asyncio.TimeoutError:
            logger.warning(f"Cache get timeout for key: {key}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode cached value for key {key}: {e}")
            # Remove corrupted cache entry
            try:
                if self.redis_client:
                    await self.redis_client.delete(key)
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: int) -> bool:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache as dictionary
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._is_available():
            return False

        try:
            if self.redis_client:
                # Serialize to JSON
                json_value = json.dumps(value, ensure_ascii=False)

                # Set with TTL and timeout
                result = await asyncio.wait_for(
                    self.redis_client.setex(key, ttl, json_value), timeout=5.0
                )

                logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")
                return bool(result)
            return False

        except asyncio.TimeoutError:
            logger.warning(f"Cache set timeout for key: {key}")
            return False
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self._is_available():
            return False

        try:
            if self.redis_client:
                result = await asyncio.wait_for(
                    self.redis_client.delete(key), timeout=5.0
                )

                logger.debug(f"Cache delete for key: {key}")
                return bool(result)
            return False

        except asyncio.TimeoutError:
            logger.warning(f"Cache delete timeout for key: {key}")
            return False
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        if not self._is_available():
            return False

        try:
            if self.redis_client:
                result = await asyncio.wait_for(
                    self.redis_client.exists(key), timeout=5.0
                )
                return bool(result)
            return False

        except asyncio.TimeoutError:
            logger.warning(f"Cache exists timeout for key: {key}")
            return False
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self._is_available():
            return {"enabled": False, "connected": False}

        try:
            if self.redis_client:
                info = await asyncio.wait_for(
                    self.redis_client.info("memory"), timeout=5.0
                )

                return {
                    "enabled": True,
                    "connected": True,
                    "memory_used": info.get("used_memory", 0),
                    "memory_used_human": info.get("used_memory_human", "0B"),
                    "max_memory": info.get("maxmemory", 0),
                    "max_memory_human": info.get("maxmemory_human", "0B"),
                }
            return {"enabled": True, "connected": False, "error": "no redis client"}

        except asyncio.TimeoutError:
            logger.warning("Cache stats timeout")
            return {"enabled": True, "connected": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}


# Global cache service instance
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """
    Get the global cache service instance.

    Returns:
        CacheService instance
    """
    return cache_service
