from emotion_detection.components.data_ingestion import DataInjection
from emotion_detection.pipeline.stage_01_DI import DataIngestionTrainingPipeline
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