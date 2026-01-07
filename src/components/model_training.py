from pathlib import Path
from typing import List, Tuple, cast

import comet_ml
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from src.components.feature_store import RedisFeatureStore
from src.config import (
    PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, COMETML_API_KEY
)
from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_data, save_object, hash_dataframe

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, 
            processed_data_path: Path | str, 
            train_data_path: Path | str, 
            test_data_path: Path | str, 
            model_path: Path | str, 
            target_name: str, 
            labels: List[str]
        ) -> None:
        self.processed_data_path = processed_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model_path = model_path
        self.target_name = target_name
        self.labels = labels

        self.feature_store = RedisFeatureStore()
        self.experiment = comet_ml.Experiment(
            api_key=COMETML_API_KEY,
            project_name="MLOps-Repo1",
            workspace="kpma1120"
        )
        self.experiment.set_name("Titanic_Model")
        self.experiment.add_tag("random_forest")
        self.experiment.add_tag("v1")

        logger.info("Model Training & Comet-ML initialized...")
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            # get all entity_ids
            entity_ids = self.feature_store.get_all_entity_ids()

            # train/test split
            train_ids, test_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

            # get batch features → DataFrame
            train_df = pd.DataFrame.from_dict(self.feature_store.get_batch_features(train_ids), orient="index")
            test_df  = pd.DataFrame.from_dict(self.feature_store.get_batch_features(test_ids), orient="index")

            # split into X / y
            X_train, y_train = train_df.drop(columns=[self.target_name]), train_df[self.target_name]
            X_test,  y_test  = test_df.drop(columns=[self.target_name]),  test_df[self.target_name]

            # save train/test data locally
            save_data(self.train_data_path, train_df)
            save_data(self.test_data_path, test_df)

            # log dataset info for processed dataset
            self.experiment.log_dataset_info(
                name="processed-dataset",
                version="1.0", 
                path=str(self.processed_data_path)  # Comet-ML accepts string only path
            )

            # log dataFrame profile and metadata for train dataset
            self.experiment.log_dataframe_profile(
                dataframe=train_df,
                name="train",
                minimal=False,
                log_raw_dataframe=True,
                dataframe_format="csv"
            )
            self.experiment.log_parameter("train_size", len(train_df))
            self.experiment.log_parameter("train_dataset_hash", hash_dataframe(train_df))

            # log dataFrame profile and metadata for test dataset
            self.experiment.log_dataframe_profile(
                dataframe=test_df,
                name="test",
                minimal=False,
                log_raw_dataframe=True,
                dataframe_format="csv"
            )
            self.experiment.log_parameter("test_size", len(test_df))
            self.experiment.log_parameter("test_dataset_hash", hash_dataframe(test_df))

            logger.info(f"Features: {X_train.columns}")
            logger.info(f"Target: {y_train.name}")
            logger.info("Preparation for Model Training completed")

            return X_train , X_test , y_train, y_test
        
        except Exception as e:
            logger.error(f"Error during Data Preparation {e}")
            raise CustomException(e)
    
    def hyperparamter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        try:
            param_distributions = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='f1', random_state=42)
            random_search.fit(X_train, y_train)

            logger.info(f"Best paramters : {random_search.best_params_}")
            return random_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Error during Hyperparamter Tuning {e}")
            raise CustomException(e)
        
    def train_and_evaluate(self , X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        try:
            best_rf: RandomForestClassifier = cast(RandomForestClassifier, self.hyperparamter_tuning(X_train, y_train))
            
            # log hyperparameters
            self.experiment.log_parameters(best_rf.get_params())

            y_pred = best_rf.predict(X_test)
            y_prob = best_rf.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)

            # log metrics
            self.experiment.log_metric("accuracy", accuracy)
            self.experiment.log_metric("precision", precision)
            self.experiment.log_metric("recall", recall)
            self.experiment.log_metric("f1_score", f1)
            self.experiment.log_metric("roc_auc", roc_auc)

            # log confusion matrix
            self.experiment.log_confusion_matrix(
                y_test.tolist(), y_pred.tolist(), labels=self.labels
            )

            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-score: {f1:.4f}")
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
            logger.info(f"Confusion Matrix:\n{cm}")

            # save model locally
            logger.info("Saving Random Forest Model...")
            save_object(self.model_path, best_rf)

            # log model artifact to Comet-ML
            self.experiment.log_model("random_forest", str(self.model_path))  # Comet-ML accepts string only path

        except Exception as e:
            logger.error(f"Error during Model Training & Evaluation {e}")
            raise CustomException(e)
        
    def run(self) -> None:
        try:
            logger.info("Starting Model Training Job....")
            X_train , X_test , y_train, y_test = self.prepare_data()
            self.train_and_evaluate(X_train , y_train, X_test , y_test)
            
            self.experiment.end()

            logger.info("End of Model Training Job")
        except Exception as e:
            logger.error(f"Error during Model Training Job {e}")
            raise CustomException(e)


if __name__ == "__main__":
    model_trainer = ModelTraining(
        PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, 
        target_name="Survived", labels=["Not Survived", "Survived"]
    )
    model_trainer.run()
