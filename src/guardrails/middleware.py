"""
Guardrails Middleware

FastAPI middleware for automatically applying enterprise guardrails
to all API requests and responses with minimal integration effort.
"""
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json

from .guardrails_orchestrator import GuardrailsOrchestrator, GuardrailResult
from .compliance_logger import ComplianceLogger
from ..config.compliance_config import get_compliance_config


class GuardrailsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic guardrails enforcement"""
    
    def __init__(self, 
                 app: FastAPI,
                 orchestrator: Optional[GuardrailsOrchestrator] = None,
                 exclude_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.orchestrator = orchestrator or GuardrailsOrchestrator()
        self.logger = logging.getLogger(__name__)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        
        # Track middleware statistics
        self.stats = {
            "requests_processed": 0,
            "violations_detected": 0,
            "requests_blocked": 0,
            "avg_processing_time_ms": 0.0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through guardrails middleware"""
        start_time = time.time()
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Extract request information
        user_id = self._extract_user_id(request)
        session_id = self._extract_session_id(request)
        
        try:
            # Process request through guardrails
            request_result = await self._process_request(request, user_id, session_id)
            
            if not request_result.is_compliant:
                # Block non-compliant requests
                self.stats["requests_blocked"] += 1
                return self._create_compliance_error_response(request_result)
            
            # Continue with normal request processing
            response = await call_next(request)
            
            # Process response through guardrails
            response_result = await self._process_response(
                request, response, user_id, session_id
            )
            
            if not response_result.is_compliant:
                # Replace response with compliant version
                response = self._create_compliant_response(response_result)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, request_result, response_result)
            
            # Add compliance headers
            self._add_compliance_headers(response, response_result)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Guardrails middleware error: {e}")
            
            # Log security alert
            self.orchestrator.compliance_logger.log_security_alert(
                alert_type="middleware_error",
                details={"error": str(e), "path": request.url.path},
                user_id=user_id
            )
            
            # Return safe error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security processing error"}
            )
    
    async def _process_request(self, 
                             request: Request, 
                             user_id: Optional[str],
                             session_id: Optional[str]) -> GuardrailResult:
        """Process incoming request through guardrails"""
        # Extract request content
        request_content = await self._extract_request_content(request)
        
        if not request_content:
            # No content to process - allow request
            return GuardrailResult(
                is_compliant=True,
                final_response="",
                confidence_score=1.0,
                violations=[],
                warnings=[],
                actions_taken=[]
            )
        
        # Process through guardrails (treating request as response)
        result = self.orchestrator.process_response(
            response_text=request_content,
            retrieved_sources=[],  # No sources for request processing
            user_id=user_id,
            session_id=session_id,
            query_text=request_content,
            context={"type": "request", "path": request.url.path}
        )
        
        return result
    
    async def _process_response(self,
                              request: Request,
                              response: Response,
                              user_id: Optional[str],
                              session_id: Optional[str]) -> GuardrailResult:
        """Process outgoing response through guardrails"""
        # Extract response content
        response_content = await self._extract_response_content(response)
        
        if not response_content:
            # No content to process - allow response
            return GuardrailResult(
                is_compliant=True,
                final_response="",
                confidence_score=1.0,
                violations=[],
                warnings=[],
                actions_taken=[]
            )
        
        # Extract sources from response metadata if available
        retrieved_sources = self._extract_sources_from_response(response)
        
        # Process through guardrails
        result = self.orchestrator.process_response(
            response_text=response_content,
            retrieved_sources=retrieved_sources,
            user_id=user_id,
            session_id=session_id,
            query_text=await self._extract_request_content(request),
            context={"type": "response", "path": request.url.path}
        )
        
        return result
    
    async def _extract_request_content(self, request: Request) -> str:
        """Extract text content from request"""
        try:
            # Handle different content types
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                body = await request.body()
                if body:
                    json_data = json.loads(body)
                    # Extract text from common JSON fields
                    text_content = self._extract_text_from_json(json_data)
                    return text_content
            
            elif "text/" in content_type:
                body = await request.body()
                return body.decode("utf-8")
            
            # Check query parameters for text content
            query_params = dict(request.query_params)
            text_fields = ["q", "query", "question", "text", "message"]
            
            for field in text_fields:
                if field in query_params:
                    return query_params[field]
            
        except Exception as e:
            self.logger.warning(f"Error extracting request content: {e}")
        
        return ""
    
    async def _extract_response_content(self, response: Response) -> str:
        """Extract text content from response"""
        try:
            # Handle StreamingResponse and other response types
            if hasattr(response, 'body'):
                body = response.body
                if isinstance(body, bytes):
                    content = body.decode("utf-8")
                    
                    # Try to parse as JSON and extract text
                    try:
                        json_data = json.loads(content)
                        return self._extract_text_from_json(json_data)
                    except json.JSONDecodeError:
                        return content
            
        except Exception as e:
            self.logger.warning(f"Error extracting response content: {e}")
        
        return ""
    
    def _extract_text_from_json(self, json_data: Any) -> str:
        """Extract text content from JSON data"""
        text_parts = []
        
        if isinstance(json_data, dict):
            # Common response fields that contain text
            text_fields = [
                "response", "answer", "result", "message", "content", 
                "text", "description", "summary", "explanation"
            ]
            
            for field in text_fields:
                if field in json_data and isinstance(json_data[field], str):
                    text_parts.append(json_data[field])
            
            # Recursively search nested objects
            for value in json_data.values():
                if isinstance(value, (dict, list)):
                    nested_text = self._extract_text_from_json(value)
                    if nested_text:
                        text_parts.append(nested_text)
        
        elif isinstance(json_data, list):
            for item in json_data:
                item_text = self._extract_text_from_json(item)
                if item_text:
                    text_parts.append(item_text)
        
        elif isinstance(json_data, str):
            text_parts.append(json_data)
        
        return " ".join(text_parts)
    
    def _extract_sources_from_response(self, response: Response) -> List[Dict]:
        """Extract source citations from response"""
        try:
            # Check response headers for source metadata
            sources_header = response.headers.get("X-RAG-Sources")
            if sources_header:
                return json.loads(sources_header)
            
            # Parse response body for source information
            if hasattr(response, 'body'):
                body = response.body
                if isinstance(body, bytes):
                    try:
                        json_data = json.loads(body.decode("utf-8"))
                        
                        # Common source fields
                        if "sources" in json_data:
                            return json_data["sources"]
                        
                        if "retrieved_documents" in json_data:
                            return json_data["retrieved_documents"]
                        
                        if "context" in json_data and isinstance(json_data["context"], list):
                            return json_data["context"]
                        
                    except json.JSONDecodeError:
                        pass
        
        except Exception as e:
            self.logger.warning(f"Error extracting sources: {e}")
        
        return []
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Check authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Would decode JWT token to get user ID
            return "user_from_token"
        
        # Check custom user header
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return user_id
        
        # Check session
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            # Would look up user from session
            return f"user_from_session_{session_id}"
        
        return None
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request"""
        return request.headers.get("X-Session-ID") or request.headers.get("Session-ID")
    
    def _create_compliance_error_response(self, result: GuardrailResult) -> JSONResponse:
        """Create error response for compliance violations"""
        config = get_compliance_config()
        
        # Different responses based on compliance level
        if config.compliance_level.value == "strict":
            message = "Request blocked due to compliance violations"
        else:
            message = "Request processed with compliance warnings"
        
        return JSONResponse(
            status_code=403 if config.compliance_level.value == "strict" else 200,
            content={
                "error": message,
                "compliance_status": "violation_detected",
                "violations": result.violations[:3],  # Limit exposed violations
                "suggestions": result.warnings[:3]
            },
            headers={
                "X-Compliance-Status": "blocked",
                "X-Violation-Count": str(len(result.violations))
            }
        )
    
    def _create_compliant_response(self, result: GuardrailResult) -> JSONResponse:
        """Create compliant response with processed content"""
        return JSONResponse(
            content={
                "response": result.final_response,
                "compliance_status": "processed",
                "confidence_score": result.confidence_score
            },
            headers={
                "X-Compliance-Status": "processed",
                "X-Confidence-Score": str(result.confidence_score)
            }
        )
    
    def _add_compliance_headers(self, response: Response, result: GuardrailResult):
        """Add compliance headers to response"""
        response.headers["X-Compliance-Processed"] = "true"
        response.headers["X-Compliance-Status"] = "compliant" if result.is_compliant else "violation"
        response.headers["X-Confidence-Score"] = str(result.confidence_score)
        
        if result.actions_taken:
            response.headers["X-Guardrails-Applied"] = ",".join(result.actions_taken[:3])
    
    def _update_stats(self, 
                     processing_time: float,
                     request_result: GuardrailResult,
                     response_result: GuardrailResult):
        """Update middleware statistics"""
        self.stats["requests_processed"] += 1
        
        if request_result.violations or response_result.violations:
            self.stats["violations_detected"] += 1
        
        # Update average processing time
        current_avg = self.stats["avg_processing_time_ms"]
        request_count = self.stats["requests_processed"]
        
        self.stats["avg_processing_time_ms"] = (
            (current_avg * (request_count - 1) + processing_time) / request_count
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            **self.stats,
            "orchestrator_stats": self.orchestrator.get_system_status()
        }


def setup_guardrails_middleware(app: FastAPI, 
                               config_path: Optional[str] = None,
                               exclude_paths: Optional[List[str]] = None) -> GuardrailsMiddleware:
    """Set up guardrails middleware on FastAPI app"""
    # Initialize orchestrator
    orchestrator = GuardrailsOrchestrator(config_path)
    
    # Create and add middleware
    middleware = GuardrailsMiddleware(
        app=app,
        orchestrator=orchestrator,
        exclude_paths=exclude_paths
    )
    
    app.add_middleware(GuardrailsMiddleware, 
                      orchestrator=orchestrator,
                      exclude_paths=exclude_paths)
    
    # Add middleware statistics endpoint
    @app.get("/guardrails/stats")
    async def get_guardrails_stats():
        """Get guardrails middleware statistics"""
        return middleware.get_statistics()
    
    @app.get("/guardrails/status")
    async def get_guardrails_status():
        """Get guardrails system status"""
        return orchestrator.get_system_status()
    
    return middleware


# Example usage
if __name__ == "__main__":
    from fastapi import FastAPI
    
    app = FastAPI(title="RAG API with Guardrails")
    
    # Set up guardrails middleware
    guardrails_middleware = setup_guardrails_middleware(
        app, 
        exclude_paths=["/health", "/docs", "/openapi.json"]
    )
    
    @app.post("/query")
    async def query_endpoint(query: dict):
        """Example RAG query endpoint"""
        # This would be processed through guardrails automatically
        return {
            "response": "This is a sample response that will be processed through guardrails",
            "sources": [
                {"id": "1", "title": "Sample Document", "content": "..."}
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint (excluded from guardrails)"""
        return {"status": "healthy"}
    
    print("FastAPI app with guardrails middleware configured")
    print("Available endpoints:")
    print("  POST /query - Main RAG endpoint with guardrails")
    print("  GET /guardrails/stats - Middleware statistics")
    print("  GET /guardrails/status - System status")
    print("  GET /health - Health check (no guardrails)")