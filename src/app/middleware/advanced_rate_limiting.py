"""
Advanced rate limiting middleware with user-based quotas and sliding windows.
"""
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict, deque
from fastapi import FastAPI, Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

from ...config.settings import settings

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        async with self.lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Remove old requests outside the window
            while (self.requests[identifier] and 
                   self.requests[identifier][0] < window_start):
                self.requests[identifier].popleft()
            
            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(current_time)
                return True
            
            return False
    
    async def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        async with self.lock:
            current_time = time.time()
            window_start = current_time - self.window_seconds
            
            # Remove old requests
            while (self.requests[identifier] and 
                   self.requests[identifier][0] < window_start):
                self.requests[identifier].popleft()
            
            return max(0, self.max_requests - len(self.requests[identifier]))
    
    async def get_reset_time(self, identifier: str) -> float:
        """Get time until rate limit resets."""
        async with self.lock:
            if not self.requests[identifier]:
                return 0
            
            oldest_request = self.requests[identifier][0]
            reset_time = oldest_request + self.window_seconds
            return max(0, reset_time - time.time())


class AdvancedRateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with multiple strategies and user-based quotas."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        
        # Different rate limiters for different scenarios
        self.global_limiter = SlidingWindowRateLimiter(
            max_requests=settings.rate_limit_requests * 10,  # Global limit
            window_seconds=settings.rate_limit_window
        )
        
        self.ip_limiter = SlidingWindowRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window
        )
        
        self.user_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        
        # User role-based limits
        self.role_limits = {
            "admin": 1000,      # Admins get higher limits
            "manager": 500,     # Managers get moderate limits
            "employee": 200,    # Employees get standard limits
            "contractor": 100,  # Contractors get lower limits
            "readonly": 50      # Read-only users get minimal limits
        }
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/v1/chat": 50,           # Chat endpoints
            "/api/v1/documents": 100,     # Document operations
            "/api/v1/ingest": 10,         # Ingestion operations (heavy)
            "/api/v1/admin": 20,          # Admin operations
            "/api/v1/analytics": 30       # Analytics endpoints
        }
        
        # Track endpoint usage
        self.endpoint_usage: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Whitelist for health checks and static assets
        self.whitelist_paths = {
            "/health", "/docs", "/redoc", "/openapi.json", 
            "/favicon.ico", "/static", "/"
        }
        
        # Suspicious activity tracking
        self.suspicious_activity: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        
    async def dispatch(self, request: Request, call_next):
        """Process request with advanced rate limiting."""
        start_time = time.time()
        
        # Skip rate limiting for whitelisted paths
        if any(request.url.path.startswith(path) for path in self.whitelist_paths):
            return await call_next(request)
        
        # Get client identifier
        client_ip = self.get_client_ip(request)
        user_id = await self.get_user_id(request)
        
        # Check if IP is temporarily blocked
        if await self.is_ip_blocked(client_ip):
            return self.create_rate_limit_response(
                "IP temporarily blocked due to suspicious activity",
                retry_after=int(self.blocked_ips[client_ip] - time.time())
            )
        
        try:
            # Global rate limiting
            if not await self.global_limiter.is_allowed("global"):
                await self.track_suspicious_activity(client_ip, "global_limit_exceeded")
                return self.create_rate_limit_response(
                    "Global rate limit exceeded",
                    retry_after=int(await self.global_limiter.get_reset_time("global"))
                )
            
            # IP-based rate limiting
            if not await self.ip_limiter.is_allowed(client_ip):
                await self.track_suspicious_activity(client_ip, "ip_limit_exceeded")
                return self.create_rate_limit_response(
                    "IP rate limit exceeded",
                    retry_after=int(await self.ip_limiter.get_reset_time(client_ip))
                )
            
            # User-based rate limiting
            if user_id:
                user_role = await self.get_user_role(request)
                if not await self.check_user_rate_limit(user_id, user_role):
                    return self.create_rate_limit_response(
                        "User rate limit exceeded",
                        retry_after=60  # Standard retry time for user limits
                    )
            
            # Endpoint-specific rate limiting
            endpoint_limit_result = await self.check_endpoint_rate_limit(
                request.url.path, client_ip, user_id
            )
            if not endpoint_limit_result["allowed"]:
                return self.create_rate_limit_response(
                    f"Endpoint rate limit exceeded",
                    retry_after=endpoint_limit_result["retry_after"]
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            await self.add_rate_limit_headers(response, client_ip, user_id)
            
            # Track request timing for anomaly detection
            request_time = time.time() - start_time
            await self.track_request_timing(client_ip, request_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Continue processing on middleware error
            return await call_next(request)
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address, considering proxies."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    async def get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from authentication token."""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            # Get access controller from app state
            access_controller = getattr(request.app.state, "access_controller", None)
            if not access_controller:
                return None
            
            token = auth_header.split(" ")[1]
            user = access_controller.validate_session(token)
            return user.user_id if user else None
            
        except Exception:
            return None
    
    async def get_user_role(self, request: Request) -> Optional[str]:
        """Get user role from authentication token."""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            access_controller = getattr(request.app.state, "access_controller", None)
            if not access_controller:
                return None
            
            token = auth_header.split(" ")[1]
            user = access_controller.validate_session(token)
            return user.role.name if user else None
            
        except Exception:
            return None
    
    async def check_user_rate_limit(self, user_id: str, user_role: str) -> bool:
        """Check user-specific rate limits based on role."""
        # Get limit for user role
        role_limit = self.role_limits.get(user_role, self.role_limits["employee"])
        
        # Create or get user limiter
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = SlidingWindowRateLimiter(
                max_requests=role_limit,
                window_seconds=settings.rate_limit_window
            )
        
        return await self.user_limiters[user_id].is_allowed(user_id)
    
    async def check_endpoint_rate_limit(self, path: str, client_ip: str, 
                                      user_id: str = None) -> Dict[str, any]:
        """Check endpoint-specific rate limits."""
        # Find matching endpoint pattern
        endpoint_limit = None
        matched_pattern = None
        
        for pattern, limit in self.endpoint_limits.items():
            if path.startswith(pattern):
                endpoint_limit = limit
                matched_pattern = pattern
                break
        
        if not endpoint_limit:
            return {"allowed": True, "retry_after": 0}
        
        # Use user_id if available, otherwise use IP
        identifier = user_id or client_ip
        
        # Check endpoint-specific limit
        current_time = time.time()
        window_start = current_time - settings.rate_limit_window
        
        # Clean old requests
        requests = self.endpoint_usage[matched_pattern][identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < endpoint_limit:
            requests.append(current_time)
            return {"allowed": True, "retry_after": 0}
        
        # Calculate retry time
        oldest_request = requests[0]
        retry_after = int((oldest_request + settings.rate_limit_window) - current_time)
        
        return {"allowed": False, "retry_after": max(1, retry_after)}
    
    async def track_suspicious_activity(self, client_ip: str, activity_type: str):
        """Track suspicious activity and block IPs if necessary."""
        current_time = time.time()
        
        # Add to suspicious activity log
        self.suspicious_activity[client_ip].append(current_time)
        
        # Clean old entries (last hour)
        hour_ago = current_time - 3600
        self.suspicious_activity[client_ip] = [
            timestamp for timestamp in self.suspicious_activity[client_ip]
            if timestamp > hour_ago
        ]
        
        # Check if IP should be blocked
        if len(self.suspicious_activity[client_ip]) > 10:  # More than 10 violations per hour
            # Block for 1 hour
            self.blocked_ips[client_ip] = current_time + 3600
            logger.warning(f"IP {client_ip} blocked for suspicious activity: {activity_type}")
    
    async def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        if client_ip in self.blocked_ips:
            if time.time() < self.blocked_ips[client_ip]:
                return True
            else:
                # Unblock expired entries
                del self.blocked_ips[client_ip]
        return False
    
    async def track_request_timing(self, client_ip: str, request_time: float):
        """Track request timing for anomaly detection."""
        # Simple anomaly detection: if request is unusually fast, it might be automated
        if request_time < 0.01:  # Less than 10ms might indicate bot activity
            await self.track_suspicious_activity(client_ip, "fast_request")
    
    async def add_rate_limit_headers(self, response, client_ip: str, user_id: str = None):
        """Add rate limit information to response headers."""
        try:
            # IP-based limits
            ip_remaining = await self.ip_limiter.get_remaining(client_ip)
            ip_reset = await self.ip_limiter.get_reset_time(client_ip)
            
            response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests)
            response.headers["X-RateLimit-Remaining"] = str(ip_remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + ip_reset))
            
            # User-based limits (if authenticated)
            if user_id and user_id in self.user_limiters:
                user_remaining = await self.user_limiters[user_id].get_remaining(user_id)
                response.headers["X-RateLimit-User-Remaining"] = str(user_remaining)
            
        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")
    
    def create_rate_limit_response(self, message: str, retry_after: int = 60):
        """Create rate limit exceeded response."""
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": message,
                "retry_after": retry_after
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Exceeded": "true"
            }
        )


def setup_advanced_rate_limiting(app: FastAPI):
    """Setup advanced rate limiting middleware."""
    app.add_middleware(AdvancedRateLimitingMiddleware)