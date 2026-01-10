import pytest
from fastapi.testclient import TestClient

from fastapi_api.api import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Verify that the root endpoint returns a welcome message with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI service"}


def test_health_endpoint() -> None:
    """Check that the health endpoint returns status ok with code 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint() -> None:
    """Ensure that the metrics endpoint returns 200 and includes prediction metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_request_total" in response.text


@pytest.mark.parametrize("payload", [
    {"PassengerId": 1, "Pclass": 3, "Name": "Test", "Sex": "male", "Age": 22,
     "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25,
     "Cabin": None, "Embarked": "S"},
])
def test_predict_success(payload: dict) -> None:
    """Test that predict endpoint returns a prediction with valid payload."""
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


@pytest.mark.parametrize("bad_payload", [
    {"PassengerId": 1, "Pclass": 3, "Name": "Test", "Sex": "male", "Age": -22,
     "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25,
     "Cabin": None, "Embarked": "S"},
    {"PassengerId": 1, "Pclass": 3, "Name": "Test", "Sex": "male", "Age": 200,
     "SibSp": 1, "Parch": 0, "Ticket": "A/5 21171", "Fare": 7.25,
     "Cabin": None, "Embarked": "S"},
    {"PassengerId": 1},
])
def test_predict_failure(bad_payload: dict) -> None:
    """Confirm that predict endpoint returns error 422 with invalid payload."""
    bad_payload = {"PassengerId": 1}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "error" in response.json()
