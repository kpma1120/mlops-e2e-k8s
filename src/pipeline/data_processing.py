from pathlib import Path

import pandas as pd

from src.components.feature_store import RedisFeatureStore
from src.components.preprocessor import Preprocessor
from src.config import PREPROCESSOR_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_data, save_data, save_object

logger = get_logger(__name__)


class DataProcessing:
    """Class for processing raw data (preprocessing and storing features in Redis)."""
    def __init__(self, 
            raw_data_path: Path | str, 
            processed_data_path: Path | str, 
            preprocessor_path: Path | str, 
            id_name: str, 
            target_name: str
        ) -> None:
        """Initialize DataProcessing with paths and identifiers.

        Args:
            raw_data_path (Path | str): Path to the raw data file.
            processed_data_path (Path | str): Path to save processed data.
            preprocessor_path (Path | str): Path to save the preprocessor object.
            id_name (str): Column name used as unique identifier.
            target_name (str): Column name of the target variable.
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.preprocessor_path = preprocessor_path
        self.id_name = id_name
        self.target_name = target_name

        self.feature_store = RedisFeatureStore()
        self.preprocessor = Preprocessor()

        logger.info("Data Processing is intialized")
        
    def store_feature_in_redis(self, df: pd.DataFrame) -> None:
        """Store processed features into Redis feature store.

        Args:
            df (pd.DataFrame): DataFrame containing processed features.

        Raises:
            CustomException: If storing features fails.
        """
        try:
            batch_data = df.set_index(self.id_name).to_dict(orient="index")
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been feeded into Feature Store")
        except Exception as e:
            logger.error(f"Error during feeding data into Feature Store {e}")
            raise CustomException(e) from e

    def run(self, start_from_processed: bool = False) -> None:
        """Run the data processing pipeline.

        This method loads raw or processed data, fits and saves the preprocessor,
        transforms the data, saves processed data, and stores features in Redis.

        Args:
            start_from_processed (bool, optional): If True, start from processed data.
                Defaults to False.

        Raises:
            CustomException: If processing fails.
        """
        try:
            logger.info("Start Data Processing...")
            
            if start_from_processed:
                df_processed = load_data(self.processed_data_path)
            else:
                df_raw = load_data(self.raw_data_path)
                self.preprocessor.fit(df_raw)
                logger.info("Saving Preprocessor...")
                save_object(self.preprocessor_path, self.preprocessor)
                logger.info("Saving Processed Data...")
                df_processed = self.preprocessor.transform(
                    df_raw, self.id_name, self.target_name
                )
                save_data(self.processed_data_path, df_processed)
            
            self.feature_store.reset()
            self.store_feature_in_redis(df_processed)

            logger.info("End of Data Processing")

        except Exception as e:
            logger.error(f"Error during Data Processing {e}")
            raise CustomException(e) from e


if __name__=="__main__":
    """Main entry point for running the data processing pipeline."""
    data_processor = DataProcessing(
        RAW_DATA_PATH, PROCESSED_DATA_PATH, PREPROCESSOR_PATH, 
        id_name="PassengerId", target_name="Survived"
    )
    data_processor.run()
