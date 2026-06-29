from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.data_ingestion import DataIngestion
from emotion_detection.logging import logger

class DataIngestionTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_files()
        data_ingestion.extract_zipfile()