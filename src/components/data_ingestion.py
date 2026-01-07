import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

from src.config import DatabaseConfigDict, DB_CONFIG, RAW_DATA_PATH
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_data

logger = get_logger(__name__)


class DataIngestion:

    def __init__(self , db_config: DatabaseConfigDict, raw_data_path: Path | str) -> None:
        self.db_config = db_config
        self.raw_data_path = raw_data_path

    def connect_to_db(self) -> Engine:
        try:
            uri = URL.create(drivername="postgresql+psycopg2", **self.db_config)
            engine = create_engine(uri)
            logger.info("Database connection established...")
            return engine
        except Exception as e:
            logger.error(f"Error during establishing connection {e}")
            raise CustomException(e)
        
    def extract_data(self) -> pd.DataFrame:
        try:
            engine = self.connect_to_db()
            query = "SELECT * FROM public.titanic"
            df = pd.read_sql_query(query,engine)
            engine.dispose()
            logger.info("Data extracted from DB")
            return df
        except Exception as e:
            logger.error(f"Error during extracting data {e}")
            raise CustomException(e)
        
    def run(self) -> None:
        try:
            logger.info("Start Data Ingestion...")
            df_raw = self.extract_data()
            logger.info("Saving Raw Data...")
            save_data(self.raw_data_path, df_raw)
            logger.info("End of Data Ingestion")
        except Exception as e:
            logger.error(f"Error during Data Ingestion {e}")
            raise CustomException(e)


if __name__=="__main__":
    data_ingestor = DataIngestion(DB_CONFIG, RAW_DATA_PATH)
    data_ingestor.run()
