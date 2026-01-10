import os

import joblib
import pandas as pd

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


def save_data(data_path, df):
    """Save a DataFrame to CSV at the specified path."""
    try:
        data_dir = os.path.dirname(data_path)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Data successfully saved at {data_path}")
    except Exception as e:
        logger.error(f"Error during saving data at {data_path}: {e}")
        raise CustomException(e) from e


def load_data(data_path):
    """Load a DataFrame from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data successfully loaded from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error during loading data from {data_path}: {e}")
        raise CustomException(e) from e


def save_object(file_path, obj):
    """Save a Python object to disk using joblib."""
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object successfully saved at {file_path}")
    except Exception as e:
        logger.error(f"Error during saving object at {file_path}: {e}")
        raise CustomException(e) from e


def load_object(file_path):
    """Load a Python object from disk using joblib."""
    try:
        obj = joblib.load(file_path)
        logger.info(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error during loading object from {file_path}: {e}")
        raise CustomException(e) from e


def hash_dataframe(df):
    """Compute a hash value for the entire DataFrame."""
    return pd.util.hash_pandas_object(df).sum()


def cast_df_float64(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all numeric columns in a DataFrame to float64."""
    df = df.copy()
    numeric_cols = df.select_dtypes(
        include=["int8","int32","int64", "float32","float64"]
    ).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")
    return df
