"""Integration tests for API endpoints."""

import pytest
import json
from fastapi.testclient import TestClient


class TestChatEndpoints:
    """Integration tests for chat endpoints."""
    
    def test_chat_endpoint_success(self, client: TestClient, api_headers):
        """Test successful chat interaction."""
        payload = {
            "message": "What is artificial intelligence?",
            "conversation_id": "test-conv-1"
        }
        
        response = client.post("/api/v1/chat", json=payload, headers=api_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
    
    def test_chat_endpoint_invalid_input(self, client: TestClient, api_headers):
        """Test chat endpoint with invalid input."""
        payload = {"invalid": "data"}
        
        response = client.post("/api/v1/chat", json=payload, headers=api_headers)
        
        assert response.status_code == 422  # Validation error


class TestDocumentEndpoints:
    """Integration tests for document endpoints."""
    
    def test_upload_document_success(self, client: TestClient, api_headers):
        """Test successful document upload."""
        files = {"file": ("test.txt", "Test document content", "text/plain")}
        
        response = client.post("/api/v1/documents", files=files, headers=api_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
    
    def test_list_documents(self, client: TestClient, api_headers):
        """Test document listing."""
        response = client.get("/api/v1/documents", headers=api_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)


class TestHealthEndpoint:
    """Integration tests for health endpoint."""
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data