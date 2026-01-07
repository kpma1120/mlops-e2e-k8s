import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()


# === Paths ===
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

RAW_DATA_PATH: Path = ARTIFACTS_DIR / "raw" / "titanic.csv"
PROCESSED_DATA_PATH: Path = ARTIFACTS_DIR / "processed" / "processed.csv"
TRAIN_DATA_PATH: Path = ARTIFACTS_DIR / "split" / "train.csv"
TEST_DATA_PATH: Path = ARTIFACTS_DIR / "split" / "test.csv"
PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "models" / "preprocessor.joblib"
MODEL_PATH: Path = ARTIFACTS_DIR / "models" / "random_forest_model.joblib"

INVALID_RAW_DATA_PATH: Path = ARTIFACTS_DIR / "invalid" / "titanic.csv"
INVALID_PREPROCESSED_DATA_PATH: Path = ARTIFACTS_DIR / "invalid" / "preprocessed.csv"
INVALID_PROCESSED_DATA_PATH: Path = ARTIFACTS_DIR / "invalid" / "processed.csv"
INVALID_RETRIEVED_DATA_PATH: Path = ARTIFACTS_DIR / "invalid" / "retrieved.csv"
DATA_VALIDATION_REPORT_PATH: Path = ARTIFACTS_DIR / "data_validation_report.json"


# === Database ===
class DatabaseConfigDict(TypedDict):
    host: str
    port: int
    username: str
    password: str
    database: str

DB_CONFIG: DatabaseConfigDict = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "username": os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_DATABASE", "postgres"),
}


# === CometML ===
COMETML_API_KEY: str = os.getenv("COMETML_API_KEY", "")
