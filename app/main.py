import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.endpoints.analysis import router as analysis_router
from app.core.connection_manager import close_connections
from app.middleware.ip_whitelist import add_ip_whitelist_middleware
from app.middleware.rate_limiter import add_rate_limiter_middleware
from app.utils.logging_config import setup_logging

# Initialize logging with DEBUG level for development
setup_logging("DEBUG")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Market TA Generator service is starting up...")
    logger.info("Prometheus metrics instrumentation enabled at /metrics")
    
    yield
    # Shutdown
    logger.info("Market TA Generator service is shutting down...")
    await close_connections()


app = FastAPI(
    title="Market TA Generator",
    description="A service that generates technical analysis for cryptocurrency pairs using LBank data and LLM",
    version="0.1.0",
    lifespan=lifespan,
)

# Initialize Prometheus metrics (must be done before adding other middleware)
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app)

# Add middleware
add_ip_whitelist_middleware(app)
add_rate_limiter_middleware(app, calls_per_minute=20, max_concurrent=3, burst_size=5)

# Include routers
app.include_router(analysis_router, prefix="/api/v1")


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    logger.debug("Health check endpoint called")
    return {"status": "ok", "service": "Market TA Generator"}
