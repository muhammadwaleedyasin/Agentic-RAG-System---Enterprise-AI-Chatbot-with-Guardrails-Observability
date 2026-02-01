"""
Enterprise FastAPI application entry point with WebSocket support and enterprise features.
"""
import logging
import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from ..config.settings import settings
from ..core.rag_pipeline import RAGPipeline
from ..models.rag import RAGQuery, RAGConfig
from ..security.access_control import AccessController, Permission
from .middleware import add_error_handlers, setup_cors, setup_logging_middleware, setup_rate_limiting
from .api import chat, documents, rag, health, conversation, security

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.value),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Global instances
rag_pipeline = RAGPipeline()
access_controller = AccessController(secret_key=settings.secret_key)
auth_scheme = HTTPBearer(auto_error=False)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str = None):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        if user_id:
            self.user_connections[user_id] = connection_id
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove from user connections
        for user_id, conn_id in list(self.user_connections.items()):
            if conn_id == connection_id:
                del self.user_connections[user_id]
                break
    
    async def send_personal_message(self, message: str, connection_id: str):
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(message)
            except:
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            connection_id = self.user_connections[user_id]
            await self.send_personal_message(message, connection_id)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(connection_id)
        
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def get_active_connections_count(self) -> int:
        """Get the count of active connections."""
        return len(self.active_connections)
    
    def get_rooms_info(self) -> dict:
        """Get information about rooms/connections."""
        return {
            "total_connections": len(self.active_connections),
            "users": list(self.user_connections.keys()),
        }

connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting up Enterprise RAG Chatbot...")
    try:
        # Initialize RAG pipeline
        await rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
        
        # Setup default admin user if none exists
        if not access_controller.users:
            admin_user = access_controller.create_user(
                user_id="admin",
                username="admin",
                email="admin@company.com",
                role_name="admin",
                password="admin123"
            )
            logger.info("Default admin user created (username: admin, password: admin123)")
        
        # Start background tasks
        asyncio.create_task(cleanup_expired_sessions())
        logger.info("Background tasks started")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Close all WebSocket connections
    for connection_id in list(connection_manager.active_connections.keys()):
        try:
            await connection_manager.active_connections[connection_id].close()
        except:
            pass
    logger.info("Application shutdown complete")


async def cleanup_expired_sessions():
    """Background task to clean up expired sessions."""
    while True:
        try:
            access_controller.cleanup_expired_sessions()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Error in session cleanup: {str(e)}")
            await asyncio.sleep(60)


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""Enterprise RAG Chatbot API with dual LLM provider support, 
    WebSocket streaming, authentication, and comprehensive management capabilities.""",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
    contact={
        "name": "Enterprise RAG Support",
        "url": "https://github.com/your-org/enterprise-rag",
        "email": "support@company.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Setup security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Setup middleware
setup_cors(app)
setup_logging_middleware(app)
setup_rate_limiting(app)
add_error_handlers(app)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = access_controller.validate_session(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

# Optional authentication dependency
async def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None
    
    return access_controller.validate_session(credentials.credentials)

# Include routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["Health"])
app.include_router(chat.router, prefix=settings.api_prefix, tags=["Chat"])
app.include_router(documents.router, prefix=settings.api_prefix, tags=["Documents"])
app.include_router(rag.router, prefix=settings.api_prefix, tags=["RAG"])
app.include_router(conversation.router, prefix=settings.api_prefix + "/conversations", tags=["Conversations"])
app.include_router(security.router, prefix=settings.api_prefix + "/security", tags=["Security"])

# Management and admin routes (protected)
try:
    from .api.routes import management, ingestion, analytics
    app.include_router(management.router, prefix=settings.api_prefix + "/admin", tags=["Management"])
    app.include_router(ingestion.router, prefix=settings.api_prefix + "/ingest", tags=["Ingestion"])
    app.include_router(analytics.router, prefix=settings.api_prefix + "/analytics", tags=["Analytics"])
except ImportError as e:
    logger.warning(f"Could not import additional routes: {e}")
    # Routes will be available once all modules are created


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "operational",
        "features": [
            "RAG Pipeline",
            "WebSocket Streaming",
            "Authentication & Authorization", 
            "Document Management",
            "Analytics & Monitoring",
            "Dual LLM Provider Support"
        ],
        "endpoints": {
            "docs": "/docs" if settings.debug else "Documentation disabled in production",
            "api": settings.api_prefix,
            "websocket": "/ws",
            "health": settings.api_prefix + "/health"
        },
        "authentication": {
            "type": "Bearer JWT",
            "login_endpoint": settings.api_prefix + "/auth/login"
        }
    }


# Authentication endpoints
@app.post(settings.api_prefix + "/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Authenticate user and return session token."""
    try:
        token = access_controller.authenticate_user(username, password)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        user = access_controller.validate_session(token)
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.name,
                "permissions": [p.value for p in user.role.permissions]
            },
            "expires_in": user.role.max_session_duration * 60  # seconds
        }
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@app.post(settings.api_prefix + "/auth/logout")
async def logout(current_user=Depends(get_current_user)):
    """Logout current user."""
    try:
        if hasattr(current_user, 'session_token') and current_user.session_token:
            access_controller.logout_user(current_user.session_token)
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return {"message": "Logout completed"}


@app.get(settings.api_prefix + "/auth/me")
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current user information."""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role.name,
        "permissions": [p.value for p in current_user.role.permissions],
        "is_active": current_user.is_active,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
        "session_expires": current_user.session_expires.isoformat() if current_user.session_expires else None
    }


# Simple user creation endpoint (temporary workaround)
@app.post(settings.api_prefix + "/admin/create-user")
async def create_user_simple(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role_name: str = Form(default="employee"),
    current_user=Depends(get_current_user)
):
    """Create a new user account (simplified endpoint)."""
    try:
        # Check admin permissions
        if not current_user.can_perform_action(Permission.ADMIN_ACCESS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        if not current_user.can_perform_action(Permission.MANAGE_USERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User management permission required"
            )
        
        # Check if username already exists
        for user in access_controller.users.values():
            if user.username == username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )
        
        # Create user
        new_user = access_controller.create_user(
            user_id=username,  # Use username as ID for simplicity
            username=username,
            email=email,
            role_name=role_name,
            password=password
        )
        
        logger.info(f"User {username} created by {current_user.username}")
        
        return {
            "status": "success",
            "message": "User created successfully",
            "user": {
                "user_id": new_user.user_id,
                "username": new_user.username,
                "email": new_user.email,
                "role": new_user.role.name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


# WebSocket endpoint for real-time chat
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, token: str = None):
    """WebSocket endpoint for real-time chat streaming."""
    # Authenticate user if token provided
    user = None
    if token:
        user = access_controller.validate_session(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    
    # Connect to manager
    await connection_manager.connect(websocket, client_id, user.user_id if user else None)
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection",
            "message": "Connected to Enterprise RAG Chatbot",
            "client_id": client_id,
            "user": user.username if user else "anonymous",
            "timestamp": time.time()
        }
        await connection_manager.send_personal_message(json.dumps(welcome_msg), client_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process different message types
            if message_data.get("type") == "chat":
                await handle_websocket_chat(message_data, client_id, user)
            elif message_data.get("type") == "ping":
                pong_msg = {"type": "pong", "timestamp": time.time()}
                await connection_manager.send_personal_message(json.dumps(pong_msg), client_id)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        connection_manager.disconnect(client_id)


async def handle_websocket_chat(message_data: dict, client_id: str, user=None):
    """Handle chat messages via WebSocket."""
    try:
        query_text = message_data.get("message", "")
        use_rag = message_data.get("use_rag", True)
        conversation_id = message_data.get("conversation_id")
        
        if not query_text.strip():
            error_msg = {
                "type": "error",
                "message": "Empty message not allowed",
                "timestamp": time.time()
            }
            await connection_manager.send_personal_message(json.dumps(error_msg), client_id)
            return
        
        # Check permissions if user authenticated
        if user and not user.can_perform_action(Permission.QUERY_SYSTEM):
            error_msg = {
                "type": "error", 
                "message": "Insufficient permissions to query system",
                "timestamp": time.time()
            }
            await connection_manager.send_personal_message(json.dumps(error_msg), client_id)
            return
        
        # Send typing indicator
        typing_msg = {
            "type": "typing",
            "message": "AI is thinking...",
            "timestamp": time.time()
        }
        await connection_manager.send_personal_message(json.dumps(typing_msg), client_id)
        
        if use_rag:
            # Process with RAG pipeline
            rag_query = RAGQuery(
                query=query_text,
                config=RAGConfig(
                    top_k=settings.retrieval_top_k,
                    similarity_threshold=settings.similarity_threshold,
                    max_context_length=settings.max_context_length
                ),
                user_id=user.user_id if user else None,
                conversation_id=conversation_id
            )
            
            rag_response = await rag_pipeline.query(rag_query)
            
            # Stream response in chunks
            response_text = rag_response.answer
            chunk_size = 50
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                is_final = i + chunk_size >= len(response_text)
                
                chunk_msg = {
                    "type": "chat_chunk",
                    "chunk": chunk,
                    "is_final": is_final,
                    "conversation_id": conversation_id,
                    "sources": rag_response.sources if is_final else None,
                    "usage": rag_response.usage if is_final else None,
                    "timestamp": time.time()
                }
                
                await connection_manager.send_personal_message(json.dumps(chunk_msg), client_id)
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
        else:
            # Direct LLM response
            response_msg = {
                "type": "chat_response",
                "message": "Direct LLM responses not yet implemented via WebSocket",
                "conversation_id": conversation_id,
                "timestamp": time.time()
            }
            await connection_manager.send_personal_message(json.dumps(response_msg), client_id)
            
    except Exception as e:
        error_msg = {
            "type": "error",
            "message": f"Error processing chat: {str(e)}",
            "timestamp": time.time()
        }
        await connection_manager.send_personal_message(json.dumps(error_msg), client_id)


def custom_openapi():
    """Custom OpenAPI schema generation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Customize schema
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add authentication schemes
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Make instances available to other modules
def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    return rag_pipeline


def get_access_controller() -> AccessController:
    """Get the global access controller instance."""
    return access_controller


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager


# Add to app state for dependency injection
app.state.rag_pipeline = rag_pipeline
app.state.access_controller = access_controller
app.state.connection_manager = connection_manager


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower()
    )
