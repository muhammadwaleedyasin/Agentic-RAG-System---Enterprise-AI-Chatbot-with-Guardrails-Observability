"""
API documentation and interactive examples.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ...config.settings import settings

router = APIRouter()

# Setup templates (would need actual template files)
templates_dir = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/docs/api", response_class=HTMLResponse)
async def api_documentation(request: Request):
    """
    Enhanced API documentation with examples.
    """
    
    api_examples = {
        "authentication": {
            "title": "Authentication",
            "description": "Authenticate and get access token",
            "examples": [
                {
                    "name": "Login",
                    "method": "POST",
                    "url": f"{settings.api_prefix}/auth/login",
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                    "body": "username=admin&password=admin123",
                    "curl": f"""curl -X POST "{settings.api_prefix}/auth/login" \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "username=admin&password=admin123\"""",
                    "response": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer",
                        "user": {
                            "user_id": "admin",
                            "username": "admin", 
                            "role": "admin",
                            "permissions": ["read_documents", "write_documents", "admin_access"]
                        },
                        "expires_in": 28800
                    }
                }
            ]
        },
        "chat": {
            "title": "Chat & RAG",
            "description": "Interact with the RAG system",
            "examples": [
                {
                    "name": "Simple Chat",
                    "method": "POST",
                    "url": f"{settings.api_prefix}/chat",
                    "headers": {
                        "Authorization": "Bearer YOUR_TOKEN",
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "message": "What is the company policy on remote work?",
                        "use_rag": True,
                        "max_tokens": 1000
                    },
                    "curl": f"""curl -X POST "{settings.api_prefix}/chat" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "What is the company policy on remote work?", "use_rag": true}}'\""",
                    "response": {
                        "status": "success",
                        "conversation_id": "conv_123",
                        "message": {
                            "role": "assistant",
                            "content": "Based on the company documentation, remote work is...",
                            "timestamp": "2024-01-01T12:00:00Z"
                        },
                        "sources": [
                            {
                                "document_id": "doc_456",
                                "similarity_score": 0.89,
                                "content_preview": "Remote work policy states..."
                            }
                        ]
                    }
                },
                {
                    "name": "Streaming Chat",
                    "method": "POST", 
                    "url": f"{settings.api_prefix}/chat/stream",
                    "headers": {
                        "Authorization": "Bearer YOUR_TOKEN",
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "message": "Explain our quarterly budget process",
                        "use_rag": True,
                        "stream": True
                    },
                    "curl": f"""curl -X POST "{settings.api_prefix}/chat/stream" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"message": "Explain our quarterly budget process", "use_rag": true}}' \\
  --no-buffer""",
                    "response": "Server-Sent Events stream with chat chunks"
                }
            ]
        },
        "documents": {
            "title": "Document Management",
            "description": "Upload and manage documents",
            "examples": [
                {
                    "name": "Upload Document",
                    "method": "POST",
                    "url": f"{settings.api_prefix}/documents/upload",
                    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                    "body": "multipart/form-data with file and metadata",
                    "curl": f"""curl -X POST "{settings.api_prefix}/documents/upload" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -F "file=@document.pdf" \\
  -F "metadata={{\\"department\\": \\"hr\\", \\"access_level\\": \\"internal\\"}}" \\
  -F "auto_process=true\"""",
                    "response": {
                        "status": "success",
                        "document_id": "doc_789",
                        "filename": "document.pdf",
                        "upload_status": "processing"
                    }
                },
                {
                    "name": "Search Documents",
                    "method": "POST",
                    "url": f"{settings.api_prefix}/documents/search",
                    "headers": {
                        "Authorization": "Bearer YOUR_TOKEN",
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "query": "employee handbook",
                        "top_k": 5,
                        "similarity_threshold": 0.7
                    },
                    "curl": f"""curl -X POST "{settings.api_prefix}/documents/search" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "employee handbook", "top_k": 5}}'\""",
                    "response": {
                        "status": "success",
                        "results": [
                            {
                                "document_id": "doc_123",
                                "similarity_score": 0.95,
                                "content": "Employee handbook section...",
                                "metadata": {"department": "hr"}
                            }
                        ]
                    }
                }
            ]
        },
        "websocket": {
            "title": "WebSocket Connection",
            "description": "Real-time chat via WebSocket",
            "examples": [
                {
                    "name": "Connect to WebSocket",
                    "method": "WebSocket",
                    "url": "ws://localhost:8000/ws/client_123?token=YOUR_TOKEN",
                    "body": "JSON messages",
                    "javascript": """
const ws = new WebSocket('ws://localhost:8000/ws/client_123?token=YOUR_TOKEN');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send a chat message
ws.send(JSON.stringify({
    type: 'chat',
    message: 'Hello, what can you help me with?',
    use_rag: true
}));
                    """,
                    "response": {
                        "type": "chat_chunk",
                        "chunk": "Hello! I can help you with...",
                        "is_final": False,
                        "timestamp": 1640995200.0
                    }
                }
            ]
        },
        "batch_ingestion": {
            "title": "Batch Ingestion",
            "description": "Upload multiple documents at once",
            "examples": [
                {
                    "name": "Batch Upload Files",
                    "method": "POST",
                    "url": f"{settings.api_prefix}/ingest/batch/upload",
                    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                    "body": "Multiple files in multipart/form-data",
                    "curl": f"""curl -X POST "{settings.api_prefix}/ingest/batch/upload" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -F "files=@doc1.pdf" \\
  -F "files=@doc2.docx" \\
  -F "files=@doc3.txt" \\
  -F "metadata={{\\"department\\": \\"legal\\"}}" \\
  -F "auto_process=true\"""",
                    "response": {
                        "status": "success",
                        "batch_id": "batch_456",
                        "total_files": 3,
                        "processing_status": "processing"
                    }
                },
                {
                    "name": "Check Batch Status",
                    "method": "GET",
                    "url": f"{settings.api_prefix}/ingest/batch/batch_456/status",
                    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                    "curl": f"""curl -X GET "{settings.api_prefix}/ingest/batch/batch_456/status" \\
  -H "Authorization: Bearer YOUR_TOKEN\"""",
                    "response": {
                        "job_id": "batch_456",
                        "status": "completed",
                        "total_files": 3,
                        "processed_files": 3,
                        "failed_files": 0
                    }
                }
            ]
        },
        "analytics": {
            "title": "Analytics & Monitoring", 
            "description": "View usage metrics and system analytics",
            "examples": [
                {
                    "name": "Usage Metrics",
                    "method": "GET",
                    "url": f"{settings.api_prefix}/analytics/usage?time_period=24h",
                    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                    "curl": f"""curl -X GET "{settings.api_prefix}/analytics/usage?time_period=24h" \\
  -H "Authorization: Bearer YOUR_TOKEN\"""",
                    "response": {
                        "time_period": "24h",
                        "total_queries": 150,
                        "total_documents": 500,
                        "avg_response_time": 1.2,
                        "queries_by_hour": {"14:00": 25, "15:00": 30}
                    }
                }
            ]
        }
    }
    
    # Return HTML documentation (simplified - would use actual template)
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{settings.app_name} - API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .example {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .code {{ background: #2d3748; color: #e2e8f0; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        .method {{ background: #3182ce; color: white; padding: 2px 8px; border-radius: 3px; }}
        .endpoint {{ color: #2d3748; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{settings.app_name} API Documentation</h1>
    <p>Version: {settings.app_version}</p>
    <p>Base URL: <code>{request.url.scheme}://{request.url.netloc}{settings.api_prefix}</code></p>
    
    <h2>Quick Start</h2>
    <ol>
        <li>Authenticate with your credentials to get an access token</li>
        <li>Include the token in the Authorization header for subsequent requests</li>
        <li>Start uploading documents and chatting with the RAG system</li>
    </ol>
    
    <h2>API Endpoints</h2>
    """
    
    for section_key, section in api_examples.items():
        html_content += f"""
        <h3>{section['title']}</h3>
        <p>{section['description']}</p>
        """
        
        for example in section['examples']:
            html_content += f"""
            <div class="example">
                <h4>{example['name']}</h4>
                <p><span class="method">{example['method']}</span> <span class="endpoint">{example['url']}</span></p>
                
                <h5>cURL Example:</h5>
                <pre class="code">{example.get('curl', 'N/A')}</pre>
                
                <h5>Response:</h5>
                <pre class="code">{str(example.get('response', 'N/A'))}</pre>
            </div>
            """
    
    html_content += """
    <h2>WebSocket Documentation</h2>
    <p>For real-time chat, connect to the WebSocket endpoint and send JSON messages.</p>
    <p>Message types: chat, ping, join_room, leave_room, direct_message</p>
    
    <h2>Rate Limits</h2>
    <p>Rate limits vary by user role:</p>
    <ul>
        <li>Admin: 1000 requests/minute</li>
        <li>Manager: 500 requests/minute</li>
        <li>Employee: 200 requests/minute</li>
        <li>Contractor: 100 requests/minute</li>
    </ul>
    
    <h2>Error Handling</h2>
    <p>All endpoints return JSON with consistent error format:</p>
    <pre class="code">{
    "status": "error",
    "message": "Error description",
    "detail": "Additional details if available"
}</pre>
    
    <h2>Authentication</h2>
    <p>Use Bearer token authentication:</p>
    <pre class="code">Authorization: Bearer YOUR_ACCESS_TOKEN</pre>
    
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@router.get("/docs/examples")
async def api_examples():
    """
    Return API examples in JSON format for programmatic access.
    """
    return {
        "version": settings.app_version,
        "base_url": settings.api_prefix,
        "examples": {
            "authentication": {
                "login": {
                    "method": "POST",
                    "endpoint": "/auth/login",
                    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                    "body": {"username": "admin", "password": "admin123"}
                }
            },
            "chat": {
                "simple_chat": {
                    "method": "POST", 
                    "endpoint": "/chat",
                    "headers": {"Authorization": "Bearer TOKEN"},
                    "body": {"message": "Hello", "use_rag": True}
                },
                "streaming_chat": {
                    "method": "POST",
                    "endpoint": "/chat/stream", 
                    "headers": {"Authorization": "Bearer TOKEN"},
                    "body": {"message": "Hello", "use_rag": True}
                }
            },
            "documents": {
                "upload": {
                    "method": "POST",
                    "endpoint": "/documents/upload",
                    "headers": {"Authorization": "Bearer TOKEN"},
                    "body": "multipart/form-data"
                },
                "search": {
                    "method": "POST",
                    "endpoint": "/documents/search",
                    "headers": {"Authorization": "Bearer TOKEN"},
                    "body": {"query": "search term", "top_k": 5}
                }
            },
            "websocket": {
                "connect": {
                    "url": "ws://host/ws/client_id?token=TOKEN",
                    "message_format": {"type": "chat", "message": "text"}
                }
            }
        }
    }


@router.get("/docs/postman")
async def postman_collection(request: Request):
    """
    Generate Postman collection for API testing.
    """
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    
    collection = {
        "info": {
            "name": f"{settings.app_name} API",
            "version": settings.app_version,
            "description": "Enterprise RAG Chatbot API Collection"
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{access_token}}", "type": "string"}]
        },
        "variable": [
            {"key": "base_url", "value": base_url},
            {"key": "api_prefix", "value": settings.api_prefix},
            {"key": "access_token", "value": ""}
        ],
        "item": [
            {
                "name": "Authentication",
                "item": [
                    {
                        "name": "Login",
                        "request": {
                            "method": "POST",
                            "header": [{"key": "Content-Type", "value": "application/x-www-form-urlencoded"}],
                            "body": {
                                "mode": "urlencoded",
                                "urlencoded": [
                                    {"key": "username", "value": "admin"},
                                    {"key": "password", "value": "admin123"}
                                ]
                            },
                            "url": {"raw": "{{base_url}}{{api_prefix}}/auth/login"}
                        }
                    }
                ]
            },
            {
                "name": "Chat",
                "item": [
                    {
                        "name": "Send Message",
                        "request": {
                            "method": "POST",
                            "header": [{"key": "Content-Type", "value": "application/json"}],
                            "body": {
                                "mode": "raw",
                                "raw": '{"message": "What is the company policy?", "use_rag": true}'
                            },
                            "url": {"raw": "{{base_url}}{{api_prefix}}/chat"}
                        }
                    }
                ]
            },
            {
                "name": "Documents", 
                "item": [
                    {
                        "name": "Upload Document",
                        "request": {
                            "method": "POST",
                            "body": {
                                "mode": "formdata",
                                "formdata": [
                                    {"key": "file", "type": "file"},
                                    {"key": "auto_process", "value": "true"}
                                ]
                            },
                            "url": {"raw": "{{base_url}}{{api_prefix}}/documents/upload"}
                        }
                    }
                ]
            }
        ]
    }
    
    return collection