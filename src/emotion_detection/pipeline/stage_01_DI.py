from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.data_ingestion import DataInjection
from emotion_detection.logging import logger

class DataIngestionTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_injection_config = config.get_data_injection_config()
        data_injection = DataInjection(config = data_injection_config)
        data_injection.download_files()
        data_injection.extract_zipfile()