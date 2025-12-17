import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from depression_prediction.constant import TARGET_COLUMN, SCHEMA_FILE_PATH
from depression_prediction.entity.config_entity import DataTransformationConfig
from depression_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from depression_prediction.exception import CustomException
from depression_prediction.logger import logging
from depression_prediction.utils.main_utils import (
    read_yaml_file,
    save_object,
    save_numpy_array_data,
    drop_columns,
)


class DataTransformation:
    """
    Builds and applies preprocessing:
    - Numeric: median impute + standard scale
    - Categorical: most_frequent impute + one-hot encode
    Saves:
    - preprocessor object
    - transformed train/test arrays (X_transformed + y as last column)
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            logging.info(f"Reading CSV: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create preprocessing object from schema YAML.
        Expects schema keys:
          - numerical_columns: [...]
          - categorical_columns: [...]
        """
        try:
            numeric_cols: List[str] = self._schema_config.get("numerical_columns", [])
            cat_cols: List[str] = self._schema_config.get("categorical_columns", [])

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {cat_cols}")

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, cat_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Reads train/test -> splits X,y -> drops schema drop_columns -> fits preprocessor on train
        -> transforms train/test -> saves arrays and preprocessor object -> returns artifact.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation...")

            preprocessor = self.get_data_transformer_object()

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in train/test data.")

            X_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_train = train_df[TARGET_COLUMN]

            X_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test = test_df[TARGET_COLUMN]

            # Optional: drop columns (e.g., IDs) if present in schema
            drop_cols = self._schema_config.get("drop_columns", [])
            if drop_cols:
                logging.info(f"Dropping columns: {drop_cols}")
                X_train_df = drop_columns(df=X_train_df, cols=drop_cols)
                X_test_df = drop_columns(df=X_test_df, cols=drop_cols)

            logging.info("Fitting preprocessor on training features and transforming train/test...")
            X_train = preprocessor.fit_transform(X_train_df)
            X_test = preprocessor.transform(X_test_df)

            # Combine X and y into arrays (y last column)
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("Saving preprocessor + transformed arrays...")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation completed: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e
