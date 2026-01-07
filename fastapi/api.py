import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from typing import List

from alibi_detect.cd import KSDrift
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.components.preprocessor import Preprocessor
from src.components.schema import inference_schema
from src.config import PROCESSED_DATA_PATH, PREPROCESSOR_PATH, MODEL_PATH
from src.logger import get_logger
from src.utils import load_data, load_object

logger = get_logger(__name__)

# API app initialization
app = FastAPI(title="Titanic Survival API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # Flask UI domain name
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model inference initialization
preprocessor: Preprocessor = load_object(PREPROCESSOR_PATH)
model: RandomForestClassifier = load_object(MODEL_PATH)

# Alibi-detect initialization
scaler = StandardScaler()


def fit_scaler_on_ref_data(
    ref_data_path: Path | str,
    scaler: StandardScaler,
    drop_columns: List[str]
) -> np.ndarray:
    """
    Fit a scaler on reference data.

    Args:
        ref_data_path: file path of reference data.
        scaler: A scikit-learn scaler instance (e.g., StandardScaler).
        drop_columns: List of column names to drop.

    Returns:
        np.ndarray: Scaled feature matrix.
    """
    
    # get reference data from file path
    df_ref = load_data(ref_data_path)

    # drop columns
    df_ref.drop(columns=drop_columns, axis=1, inplace=True)

    # fit + transform
    scaler.fit(df_ref)
    return scaler.transform(df_ref)


historical_data = fit_scaler_on_ref_data(PROCESSED_DATA_PATH, scaler, drop_columns=["PassengerId", "Survived"])
ksd = KSDrift(x_ref=historical_data, p_val=0.05)

# Prometheus initialization
prediction_total = Counter("prediction_total", "Total number of predictions")
drift_total = Counter("drift_total", "Total number of times data drift is detected")
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Model prediction latency",
    buckets=[0.02, 0.04, 0.06, 0.08, 0.1]
)
prediction_error_total = Counter("prediction_error_total", "Total number of failed predictions")


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """
    Handle prediction requests for Titanic survival inference.
    
    This endpoint accepts JSON payload with passenger data, validates and casts
    it using the inference schema, applies preprocessing transformations, performs
    drift detection, and generates a survival prediction using the trained model.

    Returns:
        Response: JSON response with prediction text if successful,
        or JSON error message with HTTP status code on failure.
    """
    try:
        start = time()
        raw_input = await request.json()
        df_raw = pd.DataFrame([raw_input])

        # validate & cast
        df_raw = inference_schema.validate(df_raw)

        # preprocess
        features = preprocessor.transform(df_raw)

        # drift detection
        features_scaled = scaler.transform(features)
        drift = ksd.predict(features_scaled)
        if drift.get("data", {}).get("is_drift", 0) == 1:
            logger.info("Drift Detected")
            drift_total.inc()

        # prediction
        prediction = model.predict(features)[0]
        prediction_total.inc()
        result = "Survived" if prediction == 1 else "Did Not Survive"

        prediction_latency_seconds.observe(time() - start)

        return JSONResponse(content={"prediction": result})
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        prediction_error_total.inc()
        return JSONResponse(content={"error": str(e)}, status_code=422)  # valid request, unprocessable content


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """
    Expose Prometheus metrics for monitoring.

    Returns:
        Response: Plain text metrics in Prometheus format.
    """
    return PlainTextResponse(generate_latest(), media_type="text/plain")


@app.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint.

    Returns:
        Response: Dict response with status ok.
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.get("/")
async def root() -> dict:
    """
    Root endpoint.

    Returns:
        Response: Dict response with welcome message.
    """
    return {"message": "Welcome to FastAPI service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
