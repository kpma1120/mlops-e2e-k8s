import os

import requests
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-service:8000")


@app.route("/health")
def health_check() -> tuple[Response, int]:
    """
    Health check endpoint.

    Returns:
        Response: JSON response with status ok.
    """
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["POST"])
def predict_proxy() -> tuple[Response, int]:
    """Proxy endpoint for forwarding prediction requests to FastAPI service.

    This Flask route receives a JSON payload from the client and forwards it
    to the FastAPI `/predict` endpoint. The response from FastAPI is returned
    to the client with the same status code. Network errors or unexpected
    exceptions are handled and returned with appropriate error codes.

    Returns:
        tuple[Response, int]: A Flask JSON response containing either the
        prediction result or an error message, along with the HTTP status code.
    """
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict", json=request.get_json(), timeout=15
        )
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 502  # unreachable
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # other unexpected errors


@app.route("/")
def index() -> str:
    """
    Root endpoint, rendering index.html.

    Returns:
        Response: Rendered template.
    """
    return render_template("index.html")


if __name__ == "__main__":
    FASTAPI_URL="http://localhost:8000"
    # Flask UI server on port 5000
    app.run(host="0.0.0.0", port=5000)
