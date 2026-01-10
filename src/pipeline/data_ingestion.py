from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from src.config import DB_CONFIG, RAW_DATA_PATH, DatabaseConfigDict
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_data

logger = get_logger(__name__)


class DataIngestion:
    """Class for ingesting data from a PostgreSQL database and saving it locally."""

    def __init__(self , 
            db_config: DatabaseConfigDict, 
            raw_data_path: Path | str
        ) -> None:
        """Initialize DataIngestion with database configuration and raw data path.

        Args:
            db_config (DatabaseConfigDict): Database connection parameters dictionary.
            raw_data_path (Path | str): Path where raw data will be saved.
        """
        self.db_config = db_config
        self.raw_data_path = raw_data_path

    def connect_to_db(self) -> Engine:
        """Establish a connection to the PostgreSQL database.

        Returns:
            Engine: SQLAlchemy Engine object for database connection.

        Raises:
            CustomException: If connection fails.
        """
        try:
            uri = URL.create(drivername="postgresql+psycopg2", **self.db_config)
            engine = create_engine(uri)
            logger.info("Database connection established...")
            return engine
        except Exception as e:
            logger.error(f"Error during establishing connection {e}")
            raise CustomException(e) from e
        
    def extract_data(self) -> pd.DataFrame:
        """Extract data from the 'titanic' table in the database.

        Returns:
            pd.DataFrame: DataFrame containing the extracted data.

        Raises:
            CustomException: If data extraction fails.
        """
        try:
            engine = self.connect_to_db()
            query = "SELECT * FROM public.titanic"
            df = pd.read_sql_query(query,engine)
            engine.dispose()
            logger.info("Data extracted from DB")
            return df
        except Exception as e:
            logger.error(f"Error during extracting data {e}")
            raise CustomException(e) from e
        
    def run(self) -> None:
        """Run the data ingestion process.

        This method extracts data from the database and saves it to the specified path.

        Raises:
            CustomException: If ingestion fails.
        """
        try:
            logger.info("Start Data Ingestion...")
            df_raw = self.extract_data()
            logger.info("Saving Raw Data...")
            save_data(self.raw_data_path, df_raw)
            logger.info("End of Data Ingestion")
        except Exception as e:
            logger.error(f"Error during Data Ingestion {e}")
            raise CustomException(e) from e


if __name__=="__main__":
    """Main entry point for running the data ingestion pipeline."""
    data_ingestor = DataIngestion(DB_CONFIG, RAW_DATA_PATH)
    data_ingestor.run()
