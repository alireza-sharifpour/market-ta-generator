"""
IP Whitelist Middleware for FastAPI.

This middleware restricts access to the API based on client IP addresses.
Only requests from whitelisted IP addresses are processed.
"""

import logging
from typing import Callable, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import WHITELIST_ENABLED, WHITELISTED_IPS

logger = logging.getLogger(__name__)


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Middleware to restrict access to the API based on IP addresses.

    Only allows requests from IP addresses in the whitelist.
    """

    def __init__(
        self,
        app: ASGIApp,
        whitelisted_ips: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the IP whitelist middleware.

        Args:
            app: The FastAPI application
            whitelisted_ips: List of IP addresses allowed to access the API.
                             If None, will use the default from config.
            enabled: Whether the whitelist is active
        """
        super().__init__(app)
        self.whitelisted_ips = whitelisted_ips or WHITELISTED_IPS
        self.enabled = enabled

        # Log the configuration
        if self.enabled:
            logger.info(
                f"IP Whitelist middleware enabled with IPs: {', '.join(self.whitelisted_ips)}"
            )
        else:
            logger.info("IP Whitelist middleware disabled")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and enforce IP whitelist if enabled.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain

        Returns:
            The response from the API or a 403 Forbidden response
        """
        # Skip whitelist check if disabled
        if not self.enabled:
            return await call_next(request)

        # Get client IP address
        client_ip = request.client.host if request.client else None

        # Check if client IP is in whitelist
        if not client_ip or client_ip not in self.whitelisted_ips:
            logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied. Your IP is not whitelisted."},
            )

        # IP is allowed, proceed with the request
        logger.debug(f"Allowed request from whitelisted IP: {client_ip}")
        return await call_next(request)


def add_ip_whitelist_middleware(app: FastAPI) -> None:
    """
    Add the IP whitelist middleware to a FastAPI application.

    Args:
        app: The FastAPI application to add the middleware to
    """
    app.add_middleware(
        IPWhitelistMiddleware,
        whitelisted_ips=WHITELISTED_IPS,
        enabled=WHITELIST_ENABLED,
    )
