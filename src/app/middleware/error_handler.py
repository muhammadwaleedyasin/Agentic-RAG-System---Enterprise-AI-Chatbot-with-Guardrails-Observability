"""
Error handling middleware for FastAPI.
"""
import logging
import traceback
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...models.common import ErrorResponse

logger = logging.getLogger(__name__)


def add_error_handlers(app: FastAPI):
    """Add error handlers to the FastAPI application."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
        
        error_response = ErrorResponse(
            error=exc.detail,
            code=exc.status_code
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle Starlette HTTP exceptions."""
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
        
        error_response = ErrorResponse(
            error=str(exc.detail),
            code=exc.status_code
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()} - {request.url}")
        
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        error_response = ErrorResponse(
            error="Validation Error",
            detail=f"Invalid request data: {error_details}",
            code=422
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        logger.error(f"Value error: {str(exc)} - {request.url}")
        
        error_response = ErrorResponse(
            error="Invalid Value",
            detail=str(exc),
            code=400
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found errors."""
        logger.error(f"File not found: {str(exc)} - {request.url}")
        
        error_response = ErrorResponse(
            error="File Not Found",
            detail=str(exc),
            code=404
        )
        
        return JSONResponse(
            status_code=404,
            content=error_response.model_dump(mode='json')
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {str(exc)} - {request.url}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        error_response = ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred. Please try again later.",
            code=500
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(mode='json')
        )