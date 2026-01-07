import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandera.errors import SchemaError

from src.components.feature_store import RedisFeatureStore
from src.components.preprocessor import Preprocessor
from src.components.schema import raw_schema, processed_schema
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, INVALID_RAW_DATA_PATH, 
    INVALID_PREPROCESSED_DATA_PATH, INVALID_PROCESSED_DATA_PATH, 
    INVALID_RETRIEVED_DATA_PATH, DATA_VALIDATION_REPORT_PATH
)
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_data, load_data


logger = get_logger(__name__)


class DataValidation:
        
        def __init__(self, 
            raw_data_path: Path | str, 
            processed_data_path: Path | str, 
            invalid_raw_data_path: Path | str, 
            invalid_preprocessed_data_path: Path | str, 
            invalid_processed_data_path: Path | str, 
            invalid_retrieved_data_path: Path | str, 
            data_validation_report_path: Path | str, 
            id_name: str, 
            target_name: str,
            test_columns: List[str]
        ) -> None:
            self.raw_data_path = raw_data_path
            self.processed_data_path = processed_data_path
            self.invalid_raw_data_path = invalid_raw_data_path
            self.invalid_preprocessed_data_path= invalid_preprocessed_data_path
            self.invalid_processed_data_path = invalid_processed_data_path
            self.invalid_retrieved_data_path = invalid_retrieved_data_path
            self.data_validation_report_path = data_validation_report_path
            self.id_name = id_name
            self.target_name = target_name
            self.test_columns = test_columns

            self.feature_store = RedisFeatureStore()

            if Path(self.raw_data_path).is_file():
                self.df_raw = load_data(self.raw_data_path)
            
            self.df_preprocessed = None

            if Path(self.processed_data_path).is_file():
                self.df_processed = load_data(self.processed_data_path)
            
            self.df_retrieved = None

            logger.info("Data Validation is intialized")

        def assert_column_equality(self, df1, df2, test_columns, df1_name="df1", df2_name="df2") -> None:
            for col in test_columns:
                s1 = df1[col].reset_index(drop=True)
                s2 = df2[col].reset_index(drop=True)

                # numeric columns (use np.allclose to allow tolerance for floating-point differences and NaN)
                if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                    assert np.allclose(s1.to_numpy(), s2.to_numpy(), equal_nan=True), \
                        f"{col} numerical values differ between {df1_name} and {df2_name}"

                # datetime columns (use exact comparison with np.array_equal)
                elif pd.api.types.is_datetime64_any_dtype(s1) and pd.api.types.is_datetime64_any_dtype(s2):
                    assert np.array_equal(s1.to_numpy(), s2.to_numpy()), \
                        f"{col} datetime values differ between {df1_name} and {df2_name}"

                # non-numeric/non-datetime columns (string, category, etc; use pandas equals for exact match)
                else:
                    assert s1.equals(s2), \
                        f"{col} values differ between {df1_name} and {df2_name}"
        
        def validate_raw(self) -> bool:
            try:
                logger.info("Start validating Raw data...")

                if not Path(self.raw_data_path).is_file():
                    logger.info("Raw data file is not yet saved, stop validating raw data")
                    valid_schema = False
                    return valid_schema

                try:
                    logger.info("Validating schema of loaded raw data...")
                    raw_schema.validate(self.df_raw)
                    valid_schema = True
                except SchemaError as e:
                    logger.error(f"Schema Validation Error in raw data: {e}")
                    save_data(self.invalid_raw_data_path, self.df_raw)  # save invalid data for debugging
                    valid_schema = False
                
                logger.info("End of Raw data validation")
                return valid_schema
            except Exception as e:
                logger.error(f"Error during Raw Data Validation {e}")
                raise CustomException(e)
        
        def validate_preprocessed(self) -> Tuple[bool, bool]:
            try:
                logger.info("Start validating Preprocessed data...")
                
                if not Path(self.raw_data_path).is_file():
                    logger.info("Raw data file is not yet saved, stop validating preprocessor transformed data")
                    valid_schema = False
                    valid_columns = False
                    return valid_schema, valid_columns

                preprocessor = Preprocessor()
                preprocessor.fit(self.df_raw)
                self.df_preprocessed = preprocessor.transform(self.df_raw, id_name=self.id_name, target_name=self.target_name)

                try:
                    logger.info("Validating schema of preprocessor transformed data...")
                    processed_schema.validate(self.df_preprocessed)
                    valid_schema = True
                except SchemaError as e:
                    logger.error(f"Schema Validation Error in preprocessed data: {e}")
                    save_data(self.invalid_preprocessed_data_path, self.df_preprocessed)  # save invalid data for debugging
                    valid_schema = False
                
                try:
                    logger.info("Validating column consistency of preprocessor transformed data...")
                    self.assert_column_equality(self.df_raw, self.df_preprocessed, self.test_columns, df1_name="raw", df2_name="preprocessed")
                    valid_columns = True
                except AssertionError as e:
                    logger.error(f"Column Consistency Test Validation Error in preprocessed data: {e}")
                    save_data(self.invalid_preprocessed_data_path, self.df_preprocessed)  # save invalid data for debugging
                    valid_columns = False
                
                logger.info("End of Preproocessed data validation")
                return valid_schema, valid_columns
            except Exception as e:
                logger.error(f"Error during Preprocessed Data Validation {e}")
                raise CustomException(e)

        def validate_processed(self) -> Tuple[bool, bool]:
            try:
                logger.info("Start validating Processed data...")

                if not Path(self.raw_data_path).is_file():
                    logger.info("Raw data file is not yet saved, stop validating processed data")
                    valid_schema = False
                    valid_columns = False
                    return valid_schema, valid_columns

                if not Path(self.processed_data_path).is_file():
                    logger.info("Processed data file is not yet saved, stop validating processed data")
                    valid_schema = False
                    valid_columns = False
                    return valid_schema, valid_columns

                try:
                    logger.info("Validating schema of loaded processed data...")
                    processed_schema.validate(self.df_processed)
                    valid_schema = True
                except SchemaError as e:
                    logger.error(f"Schema Validation Error in processed data: {e}")
                    save_data(self.invalid_processed_data_path, self.df_processed)  # save invalid data for debugging
                    valid_schema = False

                try:
                    logger.info("Validating column consistency of loaded processed data...")
                    self.assert_column_equality(self.df_raw, self.df_processed, self.test_columns, df1_name="raw", df2_name="processed")
                    valid_columns = True
                except AssertionError as e:
                    logger.error(f"Column Consistency Test Validation Error in processed data: {e}")
                    save_data(self.invalid_processed_data_path, self.df_processed)  # save invalid data for debugging
                    valid_columns = False
                    
                logger.info("End of Processed data validation")
                return valid_schema, valid_columns
            except Exception as e:
                logger.error(f"Error during Processed Data Validation {e}")
                raise CustomException(e)

        def validate_retrieved(self) -> Tuple[bool, bool]:
            try:
                logger.info("Start validating Retrieved data from feature store...")
                valid_schema = False
                valid_columns = False

                if not Path(self.processed_data_path).is_file():
                    logger.info("Processed data file is not yet saved, stop validating retrieved data")
                    return valid_schema, valid_columns

                entity_ids = self.feature_store.get_all_entity_ids()

                if len(entity_ids) == 0:
                    logger.info("Feature store is empty, stop validating retrieved data")
                    return valid_schema, valid_columns
                
                dict_retrieved = self.feature_store.get_batch_features(entity_ids)
                self.df_retrieved = pd.DataFrame.from_dict(dict_retrieved, orient="index")

                # align the index dtype to dtype of processed data id
                self.df_retrieved.index = self.df_retrieved.index.astype(self.df_processed[self.id_name].dtype)

                # order index by processed data id
                self.df_retrieved = self.df_retrieved.reindex(self.df_processed[self.id_name].tolist())

                # reset index to 1st column with id name
                self.df_retrieved.reset_index(inplace=True)
                self.df_retrieved.rename(columns={"index": self.id_name}, inplace=True)

                try:
                    logger.info("Validating schema of retrieved data...")
                    processed_schema.validate(self.df_retrieved)
                    valid_schema = True
                except SchemaError as e:
                    logger.error(f"Schema Validation Error in retrieved data: {e}")
                    save_data(self.invalid_retrieved_data_path, self.df_retrieved)  # save invalid data for debugging
                    valid_schema = False
                
                try:
                    logger.info("Validating column consistency of retrieved data...")
                    self.assert_column_equality(self.df_processed, self.df_retrieved, self.test_columns, df1_name="processed", df2_name="retrieved")
                    valid_columns = True
                except AssertionError as e:
                    logger.error(f"Column Consistency Test Validation Error in retrived data: {e}")
                    save_data(self.invalid_retrieved_data_path, self.df_retrieved)  # save invalid data for debugging
                    valid_columns = False
                
                logger.info("End of Retrieved data validation")
                return valid_schema, valid_columns
            except Exception as e:
                logger.error(f"Error during Retrieved Data Validation {e}")
                raise CustomException(e)  

        def run(self) -> None:
            try:
                logger.info("Start Data Validation...")
                valid_schema_raw = self.validate_raw()
                valid_schema_preprocessed, valid_columns_preprocessed = self.validate_preprocessed()
                valid_schema_processed, valid_columns_processed = self.validate_processed()
                valid_schema_retrieved, valid_columns_retrieved = self.validate_retrieved()
                valid_schema_overall = valid_schema_raw and valid_schema_preprocessed and valid_schema_processed and valid_schema_retrieved
                valid_columns_overall = valid_columns_preprocessed and valid_columns_processed and valid_columns_retrieved
                valid_overall_raw = valid_schema_raw
                valid_overall_preprocessed = valid_schema_preprocessed and valid_columns_preprocessed
                valid_overall_processed = valid_schema_processed and valid_columns_processed
                valid_overall_retrieved = valid_schema_retrieved and valid_columns_retrieved
                valid_overall_overall = valid_schema_overall and valid_columns_overall

                data_validation_report = {
                    "Data Validation Overall Result": f"{"Passed" if valid_overall_overall else "Failed"}",
                    "Overall Validation": {
                        "Raw Data": f"{"Passed" if valid_overall_raw else "Failed"}",
                        "Preprocessed Data": f"{"Passed" if valid_overall_preprocessed else "Failed"}",
                        "Processed Data": f"{"Passed" if valid_overall_processed else "Failed"}",
                        "Retrieved Data": f"{"Passed" if valid_overall_retrieved else "Failed"}"
                    },
                    "Schema Validation": {
                        "Overall": f"{"Passed" if valid_schema_overall else "Failed"}",
                        "Raw Data": f"{"Passed" if valid_schema_raw else "Failed"}",
                        "Preprocessed Data": f"{"Passed" if valid_schema_preprocessed else "Failed"}",
                        "Processed Data": f"{"Passed" if valid_schema_processed else "Failed"}",
                        "Retrieved Data": f"{"Passed" if valid_schema_retrieved else "Failed"}"
                    },
                    "Column Consistency Validation": {
                        "Overall": f"{"Passed" if valid_columns_overall else "Failed"}",
                        "Preprocessed Data": f"{"Passed" if valid_columns_preprocessed else "Failed"}",
                        "Processed Data": f"{"Passed" if valid_columns_processed else "Failed"}",
                        "Retrieved Data": f"{"Passed" if valid_columns_retrieved else "Failed"}"
                    }
                }
                logger.info(data_validation_report)
                with open(self.data_validation_report_path, "w") as file:
                    json.dump(data_validation_report, file)
                
                logger.info("End of Data validation")
            except Exception as e:
                logger.error(f"Error during Data Validation {e}")
                raise CustomException(e)


if __name__ == "__main__":
    data_validator = DataValidation(
        RAW_DATA_PATH, PROCESSED_DATA_PATH, 
        INVALID_RAW_DATA_PATH, INVALID_PREPROCESSED_DATA_PATH, 
        INVALID_PROCESSED_DATA_PATH, INVALID_RETRIEVED_DATA_PATH, 
        DATA_VALIDATION_REPORT_PATH, id_name="PassengerId", target_name="Survived", 
        test_columns=["PassengerId", "Survived"]
    )
    data_validator.run()
