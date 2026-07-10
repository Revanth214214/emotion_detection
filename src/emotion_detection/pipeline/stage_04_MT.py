from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.model_trainer import ModelTrainer
from emotion_detection.components.model_trainer import ModelTrainerConfig
from emotion_detection.logging import logger

class ModelTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        try:
            config_manager = configurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.initiate_model_training()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise e