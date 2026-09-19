import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sentiment-app")))

os.environ["MOCK_AUTH"] = "true"
os.environ["MOCK_PIPELINE"] = "true"
os.environ["COGNITO_CLIENT_ID"] = "mock-client-id"
os.environ["BUCKET_NAME"] = "mock-bucket"

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_login_success():
    response = client.post("/login", json={
        "username": "testuser",
        "password": "TestPassword123!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert data["tokens"]["access_token"] == "mock-access-token-xyz"

def test_login_invalid_credentials():
    response = client.post("/login", json={
        "username": "wronguser",
        "password": "wrongpassword"
    })
    assert response.status_code == 401

def test_trigger_pipeline_unauthorized():
    response = client.post("/trigger_pipeline", json={
        "filename": "test.csv",
        "user": "testuser"
    })
    assert response.status_code == 401
    assert "Missing Authorization header" in response.json()["detail"]

def test_trigger_pipeline_authorized():
    response = client.post(
        "/trigger_pipeline",
        json={"filename": "test.csv", "user": "testuser"},
        headers={"Authorization": "Bearer mock-access-token-xyz"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "pipeline_triggered"
    assert "task_arn" in response.json()

def test_generate_dashboard_unauthorized():
    response = client.post("/generate_dashboard", json={
        "filename": "test.csv",
        "user": "testuser"
    })
    assert response.status_code == 401
