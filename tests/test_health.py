"""
Health check tests for DocuLLaMA
"""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_ready_endpoint(client: TestClient):
    """Test readiness endpoint"""
    response = client.get("/ready")
    # May return 200 or 503 depending on dependencies
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_live_endpoint(client: TestClient):
    """Test liveness endpoint"""
    response = client.get("/live")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "alive"


def test_metrics_endpoint(client: TestClient):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "request_count" in data
    assert "error_count" in data
    assert "average_response_time" in data


def test_root_endpoint(client: TestClient):
    """Test root endpoint returns system info"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "DocuLLaMA"
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"