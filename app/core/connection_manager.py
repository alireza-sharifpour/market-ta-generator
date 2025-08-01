"""
Connection manager for handling HTTP clients and connection pooling.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages HTTP connections with connection pooling and best practices."""

    _instance: Optional["ConnectionManager"] = None
    _client: Optional[httpx.AsyncClient] = None

    def __new__(cls) -> "ConnectionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create an async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            # Connection pool limits for better concurrency
            limits = httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            )

            # Timeout configuration
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)

            self._client = httpx.AsyncClient(
                limits=limits, timeout=timeout, follow_redirects=True
            )

            logger.info("Created new HTTP client with connection pooling")

        return self._client

    async def close(self):
        """Close the HTTP client and clean up connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("Closed HTTP client connections")


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_http_client() -> httpx.AsyncClient:
    """Get the shared HTTP client instance."""
    return await connection_manager.get_client()


async def close_connections():
    """Close all HTTP connections. Should be called on app shutdown."""
    await connection_manager.close()
