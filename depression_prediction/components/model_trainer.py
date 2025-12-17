import sys
from typing import Dict, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from depression_prediction.logger import logging
from depression_prediction.exception import CustomException
from depression_prediction.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
)
from depression_prediction.entity.config_entity import ModelTrainerConfig
from depression_prediction.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)


class ModelTrainer:
    """
    Trains and selects the best model using transformed arrays.
    Uses class_weight='balanced' to handle imbalance (no imblearn dependency).
    Saves final pipeline: preprocessor + model
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def _split_features_target(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = arr[:, :-1]
        y = arr[:, -1]
        return X, y

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
        return ClassificationMetricArtifact(
            accuracy=accuracy_score(y_true, y_pred),
            f1_macro=f1_score(y_true, y_pred, average="macro"),
            f1_weighted=f1_score(y_true, y_pred, average="weighted"),
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        )

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.model_trainer_config.cv_folds,
            shuffle=True,
            random_state=self.model_trainer_config.random_state,
        )

    def _train_baseline(self, X_train, y_train, X_test, y_test):
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = self._compute_metrics(y_test, preds)
        return model, metrics

    def _tune_logistic_regression(self, X_train, y_train) -> GridSearchCV:
        lr = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=self.model_trainer_config.random_state,
        )
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
        }
        grid = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            scoring=self.model_trainer_config.scoring,  # e.g. "f1_macro"
            cv=self._cv(),
            n_jobs=-1,
            verbose=self.model_trainer_config.verbose,
        )
        grid.fit(X_train, y_train)
        return grid

    def _tune_random_forest(self, X_train, y_train) -> GridSearchCV:
        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.model_trainer_config.random_state,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring=self.model_trainer_config.scoring,  # e.g. "f1_macro"
            cv=self._cv(),
            n_jobs=-1,
            verbose=self.model_trainer_config.verbose,
        )
        grid.fit(X_train, y_train)
        return grid

    def _select_best(
        self,
        candidates: Dict[str, Tuple[object, ClassificationMetricArtifact]],
    ) -> Tuple[str, object, ClassificationMetricArtifact]:
        metric_name = self.model_trainer_config.selection_metric  # e.g. "f1_macro"

        best_name, best_model, best_metrics = None, None, None
        best_score = -np.inf

        for name, (model, metrics) in candidates.items():
            if not hasattr(metrics, metric_name):
                raise ValueError(f"Metric '{metric_name}' not found in ClassificationMetricArtifact.")
            score = float(getattr(metrics, metric_name))
            logging.info(f"Candidate={name} | {metric_name}={score:.4f}")

            if score > best_score:
                best_score = score
                best_name, best_model, best_metrics = name, model, metrics

        return best_name, best_model, best_metrics

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = self._split_features_target(train_arr)
            X_test, y_test = self._split_features_target(test_arr)

            logging.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

            # Baseline
            baseline_model, baseline_metrics = self._train_baseline(X_train, y_train, X_test, y_test)
            logging.info(f"Baseline metrics: {baseline_metrics}")

            # Logistic Regression (tuned)
            lr_grid = self._tune_logistic_regression(X_train, y_train)
            best_lr = lr_grid.best_estimator_
            lr_pred = best_lr.predict(X_test)
            lr_metrics = self._compute_metrics(y_test, lr_pred)
            logging.info(f"Best LR params: {lr_grid.best_params_}")
            logging.info(f"LR metrics: {lr_metrics}")

            # Random Forest (tuned)
            rf_grid = self._tune_random_forest(X_train, y_train)
            best_rf = rf_grid.best_estimator_
            rf_pred = best_rf.predict(X_test)
            rf_metrics = self._compute_metrics(y_test, rf_pred)
            logging.info(f"Best RF params: {rf_grid.best_params_}")
            logging.info(f"RF metrics: {rf_metrics}")

            # Select best
            candidates = {
                "Baseline": (baseline_model, baseline_metrics),
                "LogisticRegression": (best_lr, lr_metrics),
                "RandomForest": (best_rf, rf_metrics),
            }

            best_name, best_model, best_metrics = self._select_best(candidates)
            logging.info(f"Selected best model: {best_name}")
            logging.info(f"Best model metrics: {best_metrics}")

            # Enforce expected minimum performance
            achieved = float(getattr(best_metrics, self.model_trainer_config.selection_metric))
            if achieved < self.model_trainer_config.expected_score:
                raise Exception(
                    f"No model met expected {self.model_trainer_config.selection_metric} >= "
                    f"{self.model_trainer_config.expected_score}. Best={achieved:.4f}"
                )

            # Load preprocessor for production pipeline
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Final inference pipeline
            final_pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", best_model),
                ]
            )

            save_object(self.model_trainer_config.trained_model_file_path, final_pipeline)
            logging.info(f"Saved trained model pipeline: {self.model_trainer_config.trained_model_file_path}")

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=best_metrics,
            )

            logging.info(f"Model trainer artifact: {artifact}")
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            return artifact

        except Exception as e:
            raise CustomException(e, sys) from e
