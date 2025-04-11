import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.utils.logging_config import setup_logging

# Initialize logging
setup_logging()
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


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    logger.debug("Health check endpoint called")
    return {"status": "ok", "service": "Market TA Generator"}
