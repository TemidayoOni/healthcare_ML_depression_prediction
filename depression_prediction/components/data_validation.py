import os
import sys
import json
from dataclasses import asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from depression_prediction.constant import SCHEMA_FILE_PATH, TARGET_COLUMN
from depression_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from depression_prediction.entity.config_entity import DataValidationConfig
from depression_prediction.exception import CustomException
from depression_prediction.logger import logging
from depression_prediction.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Data Validation Component:
    - Validates schema compliance (columns present, target column exists)
    - Checks basic data quality (empty df, missing target)
    - Detects drift between train (reference) and test (current)
    - Saves drift report to configured path
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def _expected_columns(self) -> List[str]:
        """
        Expected columns from schema. Supports multiple schema formats.
        """
        columns = self._schema_config.get("columns")
        if isinstance(columns, dict):
            # schema may define {"col1": "dtype", ...}
            return list(columns.keys())
        if isinstance(columns, list):
            return columns

        # If schema does not include "columns", infer from numeric + categorical + target
        numeric_cols = self._schema_config.get("numerical_columns", [])
        cat_cols = self._schema_config.get("categorical_columns", [])
        all_cols = list(dict.fromkeys(numeric_cols + cat_cols + [TARGET_COLUMN]))
        return all_cols

    def validate_number_of_columns(self, df: DataFrame) -> bool:
        """
        Validate number of columns matches schema expectation.
        """
        try:
            expected = self._expected_columns()
            status = len(df.columns) == len(expected)
            logging.info(f"Column count check: expected={len(expected)}, actual={len(df.columns)}, status={status}")
            return status
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_required_columns_exist(self, df: DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate all required columns exist based on schema.
        Returns: (status, missing_columns)
        """
        try:
            expected = self._expected_columns()
            missing = [c for c in expected if c not in df.columns]
            status = len(missing) == 0
            if not status:
                logging.info(f"Missing required columns: {missing}")
            return status, missing
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_target_column(self, df: DataFrame) -> Tuple[bool, str]:
        """
        Validate target column existence and basic integrity.
        """
        try:
            if TARGET_COLUMN not in df.columns:
                return False, f"Target column '{TARGET_COLUMN}' not found."

            if df[TARGET_COLUMN].isna().all():
                return False, f"Target column '{TARGET_COLUMN}' is entirely missing (all NaN)."

            if df[TARGET_COLUMN].nunique(dropna=True) < 2:
                return False, f"Target column '{TARGET_COLUMN}' has <2 unique values; classification will fail."

            return True, "Target column validation passed."
        except Exception as e:
            raise CustomException(e, sys) from e

    def _simple_drift_report(
        self,
        reference_df: DataFrame,
        current_df: DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
    ) -> Dict:
        """
        Lightweight drift report without Evidently.
        - Numeric: KS test (approx via scipy if available, else summary shift)
        - Categorical: total variation distance on normalized value counts
        """
        report: Dict = {"numeric": {}, "categorical": {}, "dataset_drift": False}

        # Try KS test if scipy available
        try:
            from scipy.stats import ks_2samp  # type: ignore
            use_ks = True
        except Exception:
            use_ks = False

        drifted_features = 0
        total_features = 0

        # Numeric drift
        for col in numerical_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            total_features += 1

            ref = reference_df[col].dropna()
            cur = current_df[col].dropna()

            if ref.empty or cur.empty:
                report["numeric"][col] = {"status": "skipped_empty"}
                continue

            if use_ks:
                stat, pval = ks_2samp(ref, cur)
                drift = pval < 0.05
                report["numeric"][col] = {"ks_stat": float(stat), "p_value": float(pval), "drift": bool(drift)}
            else:
                # Fallback: compare mean/std shift
                mean_shift = float(abs(ref.mean() - cur.mean()))
                std_shift = float(abs(ref.std() - cur.std()))
                drift = (mean_shift > (0.5 * (ref.std() + 1e-9)))  # heuristic
                report["numeric"][col] = {"mean_shift": mean_shift, "std_shift": std_shift, "drift": bool(drift)}

            if report["numeric"][col].get("drift"):
                drifted_features += 1

        # Categorical drift (Total Variation Distance)
        for col in categorical_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            total_features += 1

            ref_dist = reference_df[col].astype(str).value_counts(normalize=True, dropna=False)
            cur_dist = current_df[col].astype(str).value_counts(normalize=True, dropna=False)

            # align indexes
            all_levels = ref_dist.index.union(cur_dist.index)
            ref_aligned = ref_dist.reindex(all_levels, fill_value=0.0)
            cur_aligned = cur_dist.reindex(all_levels, fill_value=0.0)

            tvd = float(0.5 * np.abs(ref_aligned - cur_aligned).sum())
            drift = tvd > 0.1  # heuristic threshold

            report["categorical"][col] = {"tvd": tvd, "drift": bool(drift)}

            if drift:
                drifted_features += 1

        report["summary"] = {
            "n_features": total_features,
            "n_drifted_features": drifted_features,
            "dataset_drift": bool(drifted_features > 0),
        }
        report["dataset_drift"] = report["summary"]["dataset_drift"]
        return report

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> Tuple[bool, Dict]:
        """
        Detect drift between reference_df (train) and current_df (test).
        Uses Evidently if available, otherwise falls back to a simple drift report.
        Returns: (drift_detected, drift_report_dict)
        """
        try:
            numerical_cols = self._schema_config.get("numerical_columns", [])
            categorical_cols = self._schema_config.get("categorical_columns", [])

            # Attempt Evidently (newer API)
            try:
                from evidently.report import Report  # type: ignore
                from evidently.metric_preset import DataDriftPreset  # type: ignore

                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=reference_df, current_data=current_df)

                report_dict = report.as_dict()
                # Evidently format varies, so we compute drift status robustly:
                drift_detected = False
                try:
                    # Many versions include a "dataset_drift" flag in the result
                    # We'll search for it
                    report_json = json.dumps(report_dict).lower()
                    drift_detected = '"dataset_drift": true' in report_json or '"dataset_drift":true' in report_json
                except Exception:
                    drift_detected = False

                return drift_detected, report_dict

            except Exception as evidently_error:
                logging.info(f"Evidently not available or failed; using simple drift check. Reason: {evidently_error}")

                report_dict = self._simple_drift_report(
                    reference_df=reference_df,
                    current_df=current_df,
                    numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols,
                )
                drift_detected = bool(report_dict.get("dataset_drift", False))
                return drift_detected, report_dict

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main entry point for Data Validation stage.
        Returns DataValidationArtifact with:
        - validation_status
        - message
        - drift_report_file_path
        """
        try:
            logging.info("Starting data validation stage.")

            validation_error_msg = []

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Basic checks: non-empty
            if train_df.empty:
                validation_error_msg.append("Training dataframe is empty.")
            if test_df.empty:
                validation_error_msg.append("Testing dataframe is empty.")

            # Required columns
            train_cols_ok, train_missing = self.validate_required_columns_exist(train_df)
            test_cols_ok, test_missing = self.validate_required_columns_exist(test_df)

            if not train_cols_ok:
                validation_error_msg.append(f"Training missing columns: {train_missing}")
            if not test_cols_ok:
                validation_error_msg.append(f"Testing missing columns: {test_missing}")

            # Target column checks
            train_target_ok, train_target_msg = self.validate_target_column(train_df)
            test_target_ok, test_target_msg = self.validate_target_column(test_df)

            if not train_target_ok:
                validation_error_msg.append(f"Train target issue: {train_target_msg}")
            if not test_target_ok:
                validation_error_msg.append(f"Test target issue: {test_target_msg}")

            validation_status = len(validation_error_msg) == 0

            drift_status = False
            drift_report = {}

            # Drift detection only if schema validation passed
            if validation_status:
                logging.info("Schema validation passed. Running drift detection.")
                drift_status, drift_report = self.detect_dataset_drift(train_df, test_df)

                # Save drift report regardless of drift status
                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=drift_report,
                )

                if drift_status:
                    logging.info("Dataset drift detected.")
                    validation_error_msg.append("Drift detected between train and test datasets.")
                else:
                    logging.info("No dataset drift detected.")

            else:
                logging.info("Skipping drift detection due to schema/target validation errors.")

            message = " | ".join(validation_error_msg) if validation_error_msg else "Validation passed."

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
