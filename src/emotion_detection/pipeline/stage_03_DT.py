from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.data_transformation import DataTransformation
from emotion_detection.logging import logger
import os

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:          
            config = configurationManager()
            data_transformation_config = config.get_data_transformation_config()

            organized_folder = data_transformation_config.organized_folder

            # Skip if the Organized folder already exists
            if os.path.exists(organized_folder):
                logger.info(
                    f"'Organized' folder already exists at {organized_folder}. "
                    "Skipping Data Transformation."
                )
                return

            logger.info("'Organized' folder not found. Running Data Transformation...")

            data_transformation = DataTransformation(config=data_transformation_config)
            
            # Run the processing loop to fill your empty folders
            data_transformation.transform_and_save_data()

            logger.info("Data Transformation completed successfully.")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise e

if __name__ == '__main__':
    # This allows you to run this specific stage standalone if needed
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()