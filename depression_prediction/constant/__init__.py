import os

from datetime import datetime

DATABASE_NAME = "DEPRESSION_DB"
COLLECTION_NAME = "data"
MONGO_DB_URL_KEY = "MONGO_URL"

PIPELINE_NAME: str = "depression_prediction"
ARTIFACTS_DIR: str = "artifacts"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

FILE_NAME = "depression.csv"
MODEL_FILE_NAME: str = "model.pkl"

TARGET_COLUMN: str = "diagnosis"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessed_object.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


"""
Data ingestion related constants

"""

DATA_INGESTION_COLLECTION_NAME: str = "data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "depressionPrediction-model2025"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 8080