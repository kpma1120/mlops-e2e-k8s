import os
import requests
from typing import Tuple

from flask import Flask, render_template, Response, jsonify, request


app = Flask(__name__)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-service:8000")


@app.route("/health")
def health_check() -> Tuple[Response, int]:
    """
    Health check endpoint.

    Returns:
        Response: JSON response with status ok.
    """
    return jsonify(status="ok"), 200


@app.route("/predict", methods=["POST"])
def predict_proxy() -> Tuple[Response, int]:
    try:
        resp = requests.post(f"{FASTAPI_URL}/predict", json=request.get_json(), timeout=15)
        resp.raise_for_status()
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 502


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

