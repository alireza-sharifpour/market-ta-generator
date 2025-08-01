"""
Rate limiting middleware to prevent API abuse and manage concurrent requests.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Callable, Dict

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiter that handles both request rate limits and concurrent request limits.
    """

    def __init__(
        self,
        app,
        calls_per_minute: int = 30,  # Max calls per minute per IP
        max_concurrent: int = 5,  # Max concurrent requests per IP
        burst_size: int = 10,  # Allow burst of requests
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.max_concurrent = max_concurrent
        self.burst_size = burst_size

        # Track request counts per IP
        self.request_counts: Dict[str, list] = defaultdict(list)
        # Track concurrent requests per IP
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        # Request queues per IP
        self.request_queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=10)
        )

        logger.info(
            f"Rate limiter initialized: {calls_per_minute} calls/min, "
            f"{max_concurrent} concurrent, {burst_size} burst"
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        if request.client:
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited."""
        current_time = time.time()
        minute_ago = current_time - 60

        # Clean old requests
        self.request_counts[client_ip] = [
            req_time
            for req_time in self.request_counts[client_ip]
            if req_time > minute_ago
        ]

        request_count = len(self.request_counts[client_ip])

        # Allow burst initially, then enforce stricter limits
        if request_count == 0:
            return False  # First request
        elif request_count <= self.burst_size:
            return False  # Within burst allowance
        else:
            return request_count >= self.calls_per_minute

    def _is_concurrent_limited(self, client_ip: str) -> bool:
        """Check if client has too many concurrent requests."""
        return self.concurrent_requests[client_ip] >= self.max_concurrent

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle rate limiting logic."""

        # Skip rate limiting for health check
        if request.url.path == "/":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Check rate limits
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 60,
                    "limits": {
                        "requests_per_minute": self.calls_per_minute,
                        "burst_size": self.burst_size,
                    },
                },
            )

        # Check concurrent request limits
        if self._is_concurrent_limited(client_ip):
            logger.warning(f"Concurrent request limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too many concurrent requests",
                    "max_concurrent": self.max_concurrent,
                    "retry_after": 10,
                },
            )

        # Track this request
        self.request_counts[client_ip].append(current_time)
        self.concurrent_requests[client_ip] += 1

        try:
            logger.debug(
                f"Processing request from {client_ip} "
                f"(concurrent: {self.concurrent_requests[client_ip]})"
            )

            response = await call_next(request)

            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.calls_per_minute - len(self.request_counts[client_ip]))
            )
            response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))

            return response

        except Exception as e:
            logger.error(f"Error processing request from {client_ip}: {str(e)}")
            raise
        finally:
            # Always decrement concurrent counter
            self.concurrent_requests[client_ip] -= 1
            if self.concurrent_requests[client_ip] <= 0:
                self.concurrent_requests[client_ip] = 0


def add_rate_limiter_middleware(app, **kwargs):
    """Add rate limiting middleware to FastAPI app."""
    app.add_middleware(RateLimiterMiddleware, **kwargs)
    logger.info("Rate limiter middleware added to application")
