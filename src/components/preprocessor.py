import numpy as np
import pandas as pd
from typing import Self

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import get_logger
from src.utils import cast_df_float64

logger = get_logger(__name__)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns) -> None:
        # ensure columns is always a list
        self.columns = columns if isinstance(columns, list) else [columns]

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()


class FamilyFeatures(BaseEstimator, TransformerMixin):
    """Compute FamilySize and IsAlone from SibSp, Parch."""

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        family_size = Xc['SibSp'] + Xc['Parch'] + 1.0
        out = pd.DataFrame({
            'FamilySize': family_size,
            'IsAlone': family_size == 1.0
        }, index=Xc.index).astype(np.float64)
        return out

class CabinFlag(BaseEstimator, TransformerMixin):
    """Binary flag for whether Cabin is present."""

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        # mapping：non-missing → 1，missing → NaN
        series = Xc['Cabin'].apply(lambda v: 1.0 if pd.notnull(v) else np.nan)
        out = pd.DataFrame({'HasCabin': series}, index=Xc.index)
        return out

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract title from Name and map to ordinal categories."""

    def __init__(self) -> None:
        # Common titles
        self.common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        # Fall back to Rare whenever unseen/missing
        self._fallback_value_ = 'Rare'

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        titles = Xc['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        out = titles.map(lambda t: t if t in self.common_titles else self._fallback_value_).to_frame()
        return out

class CombineAndInteract(BaseEstimator, TransformerMixin):
    """
    Convert outputs from ColumnTransformer into a single DataFrame with feature names,
    then combine with interaction features that rely on numeric columns.
    """

    def fit(self, X: np.ndarray, y=None) -> Self:
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        # Convert array to dataframe with feature names
        df = pd.DataFrame(data=X, columns=Preprocessor.FEATURE_NAMES[:-2])

        # Compute interactions
        interactions = pd.DataFrame({
            'Pclass_Fare': df['Pclass'] * df['Fare'],
            'Age_Fare': df['Age'] * df['Fare'],
        }, index=df.index)

        # Concatenate base + interactions
        out = pd.concat([df, interactions], axis=1).astype(np.float64)
        return out


class Preprocessor:
    """
    Encapsulates preprocessing logic for Titanic dataset using scikit-learn Pipeline.
    Can be reused in both training and inference pipelines.
    """

    FEATURE_NAMES = ['Age', 'Fare', 'Pclass', 'Sex_0', 'Sex_1', 
        'Embarked_0', 'Embarked_1', 'Embarked_2', 'FamilySize', 'IsAlone', 'HasCabin', 
        'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Title_4', 'Pclass_Fare', 'Age_Fare'
    ]

    def __init__(self) -> None:
        logger.info("Preprocessor initialized")

        # Define columns
        self.numeric_cols = ['Age', 'Fare', 'Pclass']
        self.categorical_cols = ['Sex', 'Embarked']

        # Pipelines for numeric and categorical
        numeric_pipeline = Pipeline(steps=[
            ('select', ColumnSelector(self.numeric_cols)),
            ('imputer', SimpleImputer(strategy='median')),
        ])

        categorical_pipeline = Pipeline(steps=[
            ('select', ColumnSelector(self.categorical_cols)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Handcrafted feature pipelines
        family_pipeline = Pipeline(steps=[
            ('select', ColumnSelector(['SibSp', 'Parch'])),
            ('family', FamilyFeatures()),
            ('imputer', SimpleImputer(strategy='median'))
        ])

        cabin_pipeline = Pipeline(steps=[
            ('select', ColumnSelector(['Cabin'])),
            ('cabin', CabinFlag()),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0))
        ])

        title_pipeline = Pipeline(steps=[
            ('select', ColumnSelector(['Name'])),
            ('title', TitleExtractor()),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine base features (numeric + categorical + handcrafted + identity)
        self.base_union = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_cols),
                ('cat', categorical_pipeline, self.categorical_cols),
                ('family', family_pipeline, ['SibSp', 'Parch']),
                ('cabin', cabin_pipeline, ['Cabin']),
                ('title', title_pipeline, ['Name'])
            ],
            remainder='drop'
        )

        # Final pipeline: compute base features then interactions
        self.pipeline = Pipeline(steps=[
            ('base', self.base_union),
            # Interaction needs Pclass, Age, Fare in single frame → we implement a wrapper
            ('combine_and_interact', CombineAndInteract())
        ])

    def fit(self,
            df: pd.DataFrame
        )  -> Self:
        """
        Fit the preprocessing pipeline to training data.
        This will fix parameters (e.g., median/mode, encoders).
        """
        try:
            logger.info("Fitting preprocessing pipeline")
            # Fit pipeline on raw dataframe (it handles selection internally)
            df = cast_df_float64(df)
            self.pipeline.fit(df)
            logger.info("Preprocessing pipeline fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error during preprocessing fit: {e}")
            raise CustomException(e)

    def transform(self,
            df: pd.DataFrame,
            id_name: str | None = None,
            target_name: str | None = None
        ) -> pd.DataFrame:
        """
        Apply preprocessing transformations to raw Titanic dataframe.
        Returns a processed dataframe with engineered features, and optional key and target columns.
        Signature preserved.
        """
        try:
            df = df.copy()

            # Transform through pipeline
            features = self.pipeline.transform(cast_df_float64(df))

            # Ensure output is DataFrame with index aligned
            if isinstance(features, pd.DataFrame):
                df_transformed = features.copy()
            else:
                # ColumnTransformer/Pipeline may output numpy; convert to DataFrame
                df_transformed = pd.DataFrame(features, index=df.index)

            # Attach target if requested
            if target_name:
                df_transformed = pd.concat([df[[target_name]], df_transformed], axis=1)

            # Attach id if requested
            if id_name:
                df_transformed = pd.concat([df[[id_name]], df_transformed], axis=1)

            logger.info("Preprocessing transformations applied successfully")
            return df_transformed

        except Exception as e:
            logger.error(f"Error during preprocessing transform: {e}")
            raise CustomException(e)
