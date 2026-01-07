from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)


@app.route("/health")
def health_check() -> Response:
    """
    Health check endpoint.

    Returns:
        Response: JSON response with status ok.
    """
    return jsonify(status="ok")


@app.route("/")
def index() -> str:
    """
    Root endpoint, rendering index.html.

    Returns:
        Response: Rendered template.
    """
    return render_template("index.html")


if __name__ == "__main__":
    # Flask UI server on port 5000
    app.run(host="0.0.0.0", port=5000)
