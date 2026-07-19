from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.data_transformation import DataTransformation
from emotion_detection.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = configurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            
            # Run the processing loop to fill your empty folders
            data_transformation.transform_and_save_data()
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise e

if __name__ == '__main__':
    # This allows you to run this specific stage standalone if needed
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()