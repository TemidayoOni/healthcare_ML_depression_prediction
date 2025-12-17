import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from depression_prediction.entity.config_entity import DataIngestionConfig
from depression_prediction.entity.artifact_entity import DataIngestionArtifact
from depression_prediction.exception import CustomException
from depression_prediction.logger import logging
from depression_prediction.data_access.data import Data


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as pandas dataframe into feature store
        """
        try:
            logging.info("Exporting data from mongodb to feature store")
            data = Data()
            dataframe = data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"shape of dataframe: {dataframe.shape}")

            # create feature store directory if not available
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)
        

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Split the data into train and test set
        Description: This method splits the input dataframe into training and testing sets based on the configuration provided during initialization. It saves the resulting datasets to specified file paths.
        Output: folder is created in s3 bucket
        On Failure: raises CustomException
        """
        logging.info("Entered the split_data_as_train_test method of DataIngestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio)
            
            logging.info("Performed train test split on the dataframe")
            logging.info("exited split_data_as_train_test method of DataIngestion class")
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info("Exported train and test file path.")
        except Exception as e:
            raise CustomException(e, sys) from e
        


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process
        """
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got data from mongodb")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed treain test split on the dataset")

            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")


            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    

    