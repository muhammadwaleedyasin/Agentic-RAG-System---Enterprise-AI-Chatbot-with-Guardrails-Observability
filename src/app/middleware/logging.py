"""
Logging middleware for FastAPI.
"""
import time
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.settings import settings

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request/response information."""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log information."""
        start_time = time.time()
        
        # Log request
        logger.info(f"{request.method} {request.url} - Start")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"{request.method} {request.url} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


def setup_logging_middleware(app: FastAPI):
    """Setup logging middleware for the FastAPI application."""
    
    if settings.debug:
        app.add_middleware(LoggingMiddleware)