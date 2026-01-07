# This import is needed to resolve COMET WARNING: To get all data logged automatically, 
# import comet_ml before the following modules: sklearn.
import comet_ml  

from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing
from src.components.model_training import ModelTraining
from src.config import (
    DB_CONFIG, RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, 
    PREPROCESSOR_PATH, MODEL_PATH
)

if __name__=="__main__":
    data_ingestor = DataIngestion(DB_CONFIG, RAW_DATA_PATH)
    data_ingestor.run()

    data_processor = DataProcessing(
        RAW_DATA_PATH, PROCESSED_DATA_PATH, PREPROCESSOR_PATH, 
        id_name="PassengerId", target_name="Survived"
    )
    data_processor.run()

    model_trainer = ModelTraining(
        PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, 
        target_name="Survived", labels=["Not Survived", "Survived"]
    )
    model_trainer.run()
