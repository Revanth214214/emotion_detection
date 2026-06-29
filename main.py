from emotion_detection.components.data_ingestion import DataIngestion
from emotion_detection.pipeline.stage_01_DI import DataIngestionTrainingPipeline
from emotion_detection.components.data_validation import Datavalidation
from emotion_detection.pipeline.stage_02_DV import DataValidationTrainingPipeline
from emotion_detection.components.data_transformation import DataTransformation
from emotion_detection.pipeline.stage_03_DT import DataTransformationTrainingPipeline
from emotion_detection.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e