import sys

from depression_prediction.exception import CustomException
from depression_prediction.logger import logging

import os
from depression_prediction.constant import DATABASE_NAME, MONGO_DB_URL_KEY
import pymongo
import certifi

ca = certifi.where()

class MongoDBClient:

    """
    Class Name:  export_data_into_feature_store
    Description:  This method exports the dataframe from mongodb feature store as dataframe

    Output:  connection to mongodb database
    On Failure: Raise CustomException
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGO_DB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment variable: {MONGO_DB_URL_KEY} not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Connected to MongoDB database: {database_name}")
        except Exception as e:
            raise CustomException(e, sys)
