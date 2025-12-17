import os
import sys
import shutil

from depression_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from depression_prediction.entity.config_entity import ModelPusherConfig
from depression_prediction.exception import CustomException
from depression_prediction.logger import logging

# from your_project.cloud_storage.aws_storage import SimpleStorageService
# from your_project.entity.s3_estimator import ProjectEstimator


class ModelPusher:
    """
    ModelPusher:
    - Pushes (promotes) the trained model to a "best model" location if accepted.
    - Optionally uploads the accepted model to S3 (if configured).
    """

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        :param model_evaluation_artifact: Output of Model Evaluation stage
        :param model_pusher_config: Configuration for pushing model (local + optional s3)
        """
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

            # Optional AWS client
            # self.s3 = SimpleStorageService()

        except Exception as e:
            raise CustomException(e, sys) from e

    def _promote_model_locally(self) -> str:
        """
        Copies the trained model to the 'best model' path (local promotion).
        Returns the destination path.
        """
        trained_model_path = self.model_evaluation_artifact.trained_model_path
        best_model_path = self.model_pusher_config.best_model_path

        if not trained_model_path or not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Trained model not found at: {trained_model_path}")

        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        shutil.copy2(trained_model_path, best_model_path)
        logging.info(f"Promoted model locally from {trained_model_path} -> {best_model_path}")

        return best_model_path

    def _push_to_s3(self) -> str:
        """
        OPTIONAL:
        Upload model to S3 if you have AWS infrastructure.
        Returns the s3 path/key.
        """
        bucket_name = self.model_pusher_config.bucket_name
        s3_model_key_path = self.model_pusher_config.s3_model_key_path
        trained_model_path = self.model_evaluation_artifact.trained_model_path

        if not bucket_name or not s3_model_key_path:
            raise ValueError("S3 bucket_name or s3_model_key_path missing in ModelPusherConfig.")

        if not trained_model_path or not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Trained model not found at: {trained_model_path}")

        # If you have an estimator wrapper:
        # estimator = ProjectEstimator(bucket_name=bucket_name, model_path=s3_model_key_path)
        # estimator.save_model(from_file=trained_model_path)

        # Or using a generic storage service:
        # self.s3.upload_file(
        #     from_file=trained_model_path,
        #     to_filename=s3_model_key_path,
        #     bucket_name=bucket_name,
        # )

        logging.info(f"(Stub) Uploaded model to S3 bucket={bucket_name}, key={s3_model_key_path}")
        return s3_model_key_path

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Main entry point.

        If model is accepted:
          - promote locally to best_model_path
          - optionally upload to S3
        If not accepted:
          - do nothing (or still return artifact with info)
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model was NOT accepted. Skipping model push.")
                return ModelPusherArtifact(
                    bucket_name=getattr(self.model_pusher_config, "bucket_name", ""),
                    s3_model_path=getattr(self.model_pusher_config, "s3_model_key_path", ""),
                )

            logging.info("Model accepted. Proceeding to push/promote model.")

            # 1) Promote locally (recommended)
            promoted_path = self._promote_model_locally()

            # 2) Optional: push to S3
            if getattr(self.model_pusher_config, "enable_s3_push", False):
                self._push_to_s3()

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=getattr(self.model_pusher_config, "bucket_name", ""),
                s3_model_path=getattr(self.model_pusher_config, "s3_model_key_path", promoted_path),
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
