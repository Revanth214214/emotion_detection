from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.data_transformation import DataTransformation
from emotion_detection.logging import logger

class DataTransformationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config_manager = configurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.verify_and_prepare_data()