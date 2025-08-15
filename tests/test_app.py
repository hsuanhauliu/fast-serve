import asyncio
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

# Assuming your create_app function is in a file named `app.py`
from fast_serve.app import create_app


# --- Test Setup ---


# Define Pydantic models for testing request and response
class TestInput(BaseModel):
    text: str


class TestOutput(BaseModel):
    prediction: str


# Define synchronous and asynchronous prediction functions for testing
def sync_predict(data: TestInput) -> TestOutput:
    """A simple synchronous prediction function for tests."""
    return TestOutput(prediction=f"Sync prediction for: {data.text}")


async def async_predict(data: TestInput) -> TestOutput:
    """A simple asynchronous prediction function for tests."""
    await asyncio.sleep(0.01)  # Simulate a small I/O delay
    return TestOutput(prediction=f"Async prediction for: {data.text}")


# --- Pytest Fixtures for Different App Configurations ---


@pytest.fixture
def sync_http_app():
    """Provides a TestClient for an app with a sync HTTP endpoint."""
    app = create_app(
        predict_func=sync_predict,
        response_model=TestOutput,
        http_endpoint="/predict",
        websocket_endpoint=None,
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def async_http_app():
    """Provides a TestClient for an app with an async HTTP endpoint."""
    app = create_app(
        predict_func=async_predict,
        response_model=TestOutput,
        http_endpoint="/predict",
        websocket_endpoint=None,
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sync_websocket_app():
    """Provides a TestClient for an app with a sync WebSocket endpoint."""
    app = create_app(
        predict_func=sync_predict,
        http_endpoint=None,
        websocket_endpoint="/ws",
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def async_websocket_app():
    """Provides a TestClient for an app with an async WebSocket endpoint."""
    app = create_app(
        predict_func=async_predict,
        http_endpoint=None,
        websocket_endpoint="/ws",
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def full_app():
    """Provides a TestClient for an app with both async HTTP and WebSocket endpoints."""
    app = create_app(
        predict_func=async_predict,
        response_model=TestOutput,
        http_endpoint="/predict",
        websocket_endpoint="/ws",
    )
    with TestClient(app) as client:
        yield client


# --- Test Cases ---


def test_sync_http_endpoint(sync_http_app):
    """Tests the HTTP endpoint with a synchronous prediction function."""
    response = sync_http_app.post("/predict", json={"text": "hello"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "Sync prediction for: hello"}


def test_async_http_endpoint(async_http_app):
    """Tests the HTTP endpoint with an asynchronous prediction function."""
    response = async_http_app.post("/predict", json={"text": "world"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "Async prediction for: world"}


def test_sync_websocket_endpoint(sync_websocket_app):
    """Tests the WebSocket endpoint with a synchronous prediction function."""
    with sync_websocket_app.websocket_connect("/ws") as websocket:
        websocket.send_json({"text": "hello ws"})
        data = websocket.receive_json()
        assert data == {"prediction": "Sync prediction for: hello ws"}


def test_async_websocket_endpoint(async_websocket_app):
    """Tests the WebSocket endpoint with an asynchronous prediction function."""
    with async_websocket_app.websocket_connect("/ws") as websocket:
        websocket.send_json({"text": "world ws"})
        data = websocket.receive_json()
        assert data == {"prediction": "Async prediction for: world ws"}


def test_full_app_endpoints(full_app):
    """Tests an app with both HTTP and WebSocket endpoints enabled."""
    # Test HTTP
    response = full_app.post("/predict", json={"text": "full http"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "Async prediction for: full http"}

    # Test WebSocket
    with full_app.websocket_connect("/ws") as websocket:
        websocket.send_json({"text": "full ws"})
        data = websocket.receive_json()
        assert data == {"prediction": "Async prediction for: full ws"}


def test_http_endpoint_disabled():
    """Tests that the HTTP endpoint is not created when set to None."""
    app = create_app(predict_func=sync_predict, http_endpoint=None, websocket_endpoint="/ws")
    client = TestClient(app)
    response = client.post("/predict", json={"text": "test"})
    assert response.status_code == 404  # Not Found


def test_websocket_endpoint_disabled():
    """Tests that the WebSocket endpoint is not created when set to None."""
    app = create_app(predict_func=sync_predict, websocket_endpoint=None)
    client = TestClient(app)
    with pytest.raises(Exception):  # Exact exception can vary
        with client.websocket_connect("/ws"):
            pass  # Should not connect

def test_raises_error_if_no_endpoints():
    """Tests that a ValueError is raised if both endpoints are None."""
    with pytest.raises(ValueError):
        create_app(predict_func=sync_predict, http_endpoint=None, websocket_endpoint=None)
