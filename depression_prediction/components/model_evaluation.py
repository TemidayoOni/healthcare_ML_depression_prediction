import sys
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from depression_prediction.constant import TARGET_COLUMN

from depression_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact,
)
from depression_prediction.entity.config_entity import ModelEvaluationConfig
from depression_prediction.exception import CustomException
from depression_prediction.logger import logging
from depression_prediction.utils.main_utils import load_object

# OPTIONAL: If you have an S3 estimator wrapper like the visa example, import it
# from your_project.entity.s3_estimator import ProjectEstimator


@dataclass
class EvaluateModelResponse:
    trained_metric: float
    best_metric: float
    is_model_accepted: bool
    difference: float
    metric_name: str


class ModelEvaluation:
    """
    ModelEvaluation compares the newly trained model with the currently deployed/best model.
    It decides whether to accept the newly trained model.

    - Reads test dataset from DataIngestionArtifact
    - Evaluates trained model (from ModelTrainerArtifact)
    - Loads best/production model (if available) and evaluates it
    - Compares using a selection metric (default: f1_macro)
    """

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def _compute_metrics(self, y_true, y_pred) -> ClassificationMetricArtifact:
        """
        Computes the same metrics used in your training notebook:
        - accuracy
        - f1_macro
        - f1_weighted
        - precision_macro
        - recall_macro
        """
        return ClassificationMetricArtifact(
            accuracy=accuracy_score(y_true, y_pred),
            f1_macro=f1_score(y_true, y_pred, average="macro"),
            f1_weighted=f1_score(y_true, y_pred, average="weighted"),
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        )

    def _get_selection_metric_value(self, metrics: ClassificationMetricArtifact) -> float:
        """
        Returns the metric used to decide whether the model is accepted.
        Configurable from ModelEvaluationConfig (default: 'f1_macro').
        """
        metric_name = getattr(self.model_eval_config, "selection_metric", "f1_macro")
        if not hasattr(metrics, metric_name):
            raise ValueError(
                f"selection_metric '{metric_name}' not found in ClassificationMetricArtifact."
            )
        return float(getattr(metrics, metric_name))

    def get_best_model(self):
        """
        Get best/production model.

        Option A (recommended for a local project):
          - Load from local path in config: model_eval_config.best_model_path

        Option B (cloud deployment):
          - Load from S3 (if you have estimator wrapper)

        This implementation supports Option A out of the box.
        """
        try:
            best_model_path = getattr(self.model_eval_config, "best_model_path", None)
            if best_model_path and isinstance(best_model_path, str):
                try:
                    best_model = load_object(best_model_path)
                    logging.info(f"Loaded best model from: {best_model_path}")
                    return best_model
                except Exception as e:
                    logging.info(f"No best model found at {best_model_path}. Reason: {e}")
                    return None

            # If you want S3-based loading, uncomment and implement using your S3 estimator:
            # bucket_name = self.model_eval_config.bucket_name
            # model_key_path = self.model_eval_config.s3_model_key_path
            # estimator = ProjectEstimator(bucket_name=bucket_name, model_path=model_key_path)
            # if estimator.is_model_present(model_path=model_key_path):
            #     return estimator
            # return None

            return None

        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate trained model against best model using the configured selection metric.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            if TARGET_COLUMN not in test_df.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test data.")

            X_test = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test = test_df[TARGET_COLUMN]

            # Load trained model (full pipeline)
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)

            # Predict and compute metrics for trained model
            y_hat_trained = trained_model.predict(X_test)
            trained_metrics = self._compute_metrics(y_test, y_hat_trained)

            trained_score = self._get_selection_metric_value(trained_metrics)
            selection_metric_name = getattr(self.model_eval_config, "selection_metric", "f1_macro")

            # Compare with best model if it exists
            best_model = self.get_best_model()
            if best_model is not None:
                # If best_model is an estimator wrapper, you might need best_model.predict(X_test)
                y_hat_best = best_model.predict(X_test)
                best_metrics = self._compute_metrics(y_test, y_hat_best)
                best_score = self._get_selection_metric_value(best_metrics)
            else:
                best_score = 0.0  # No production model exists yet

            is_accepted = trained_score > best_score
            difference = trained_score - best_score

            logging.info(f"Trained {selection_metric_name}: {trained_score:.4f}")
            logging.info(f"Best {selection_metric_name}: {best_score:.4f}")
            logging.info(f"Model accepted: {is_accepted}, Difference: {difference:.4f}")

            return EvaluateModelResponse(
                trained_metric=float(trained_score),
                best_metric=float(best_score),
                is_model_accepted=bool(is_accepted),
                difference=float(difference),
                metric_name=selection_metric_name,
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiate evaluation and return ModelEvaluationArtifact.
        """
        try:
            eval_response = self.evaluate_model()

            # If you use S3, you can keep these fields meaningful.
            # Otherwise, you can store a local "best model path" in s3_model_path.
            s3_or_best_path = getattr(self.model_eval_config, "s3_model_key_path", None)
            if not s3_or_best_path:
                s3_or_best_path = getattr(self.model_eval_config, "best_model_path", "")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                changed_accuracy=eval_response.difference,
                s3_model_path=s3_or_best_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
