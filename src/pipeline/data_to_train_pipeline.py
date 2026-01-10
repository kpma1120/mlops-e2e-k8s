# This import is needed to resolve COMET WARNING: To get all data logged automatically, 
# import comet_ml before the following modules: sklearn.
import comet_ml

from src.config import (
    DB_CONFIG,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.logger import get_logger
from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.data_processing import DataProcessing
from src.pipeline.model_training import ModelTraining

logger = get_logger(__name__)


if __name__=="__main__":
    """Main entry point for the end-to-end ML pipeline.

    This script orchestrates the workflow by running data ingestion, data processing,
    and model training sequentially. It ensures raw data is ingested from the database,
    preprocessed and stored, and then used to train and evaluate a Random Forest model
    with Comet-ML integration for experiment tracking.
    """
    data_ingestor = DataIngestion(DB_CONFIG, RAW_DATA_PATH)
    data_ingestor.run()

    data_processor = DataProcessing(
        RAW_DATA_PATH, PROCESSED_DATA_PATH, PREPROCESSOR_PATH, 
        id_name="PassengerId", target_name="Survived"
    )
    data_processor.run()

    logger.info(f"Comet-ML version:{comet_ml.__version__} used for model tracking")

    model_trainer = ModelTraining(
        PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, 
        target_name="Survived", labels=["Not Survived", "Survived"]
    )
    model_trainer.run()
