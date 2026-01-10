import pytest
import requests
from flask.testing import FlaskClient

from flask_ui.ui import app


@pytest.fixture
def client() -> FlaskClient:
    """Provide a Flask test client configured for testing."""
    app.config["TESTING"] = True
    return app.test_client()


def test_root_endpoint(client: FlaskClient) -> None:
    """Verify that the root endpoint returns HTML content with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.get_data(as_text=True)


def test_health_endpoint(client: FlaskClient) -> None:
    """Check that the health endpoint returns status ok with code 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_proxy_success(
        monkeypatch: pytest.MonkeyPatch, 
        client: FlaskClient
    ) -> None:
    """Ensure predict proxy returns a successful prediction when FastAPI responds."""
    class DummyResponse:
        def __init__(self):
            self.status_code = 200
        def json(self):
            return {"prediction": "Survived"}
        def raise_for_status(self):
            return None
    monkeypatch.setattr("requests.post", lambda *a, **kw: DummyResponse())

    response = client.post("/predict", json={"PassengerId": 1})
    assert response.status_code == 200
    assert response.get_json() == {"prediction": "Survived"}


def test_predict_proxy_failure(
        monkeypatch: pytest.MonkeyPatch, 
        client: FlaskClient
    ) -> None:
    """Confirm predict proxy returns error 502 when FastAPI is unreachable."""
    def dummy_post(*a, **kw):
        raise requests.exceptions.RequestException("FastAPI not reachable")
    monkeypatch.setattr("requests.post", dummy_post)

    response = client.post("/predict", json={"PassengerId": 1})
    assert response.status_code == 502
    assert "error" in response.get_json()
