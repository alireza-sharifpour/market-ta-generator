import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.endpoints.analysis import router as analysis_router
from app.middleware.ip_whitelist import add_ip_whitelist_middleware
from app.utils.logging_config import setup_logging

# Initialize logging with DEBUG level for development
setup_logging("DEBUG")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Market TA Generator service is starting up...")
    yield
    # Shutdown
    logger.info("Market TA Generator service is shutting down...")


app = FastAPI(
    title="Market TA Generator",
    description="A service that generates technical analysis for cryptocurrency pairs using LBank data and LLM",
    version="0.1.0",
    lifespan=lifespan,
)

# Add middleware
add_ip_whitelist_middleware(app)

# Include routers
app.include_router(analysis_router, prefix="/api/v1")


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    logger.debug("Health check endpoint called")
    return {"status": "ok", "service": "Market TA Generator"}
