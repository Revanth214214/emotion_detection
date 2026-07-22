from emotion_detection.config.configuration import configurationManager
from emotion_detection.components.model_trainer import ModelTrainer
from emotion_detection.logging import logger
import os
import urllib.request

STAGE_NAME = "Model Trainer Stage"

# URL of the pretrained model
MODEL_URL = "https://github.com/San0160/emotion_detection/raw/refs/heads/main/models/best_rafdb_resnet18.pth"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_path = model_trainer_config.model_path

        # If model already exists, skip everything
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}. Skipping training.")
            return
        
        # Try downloading the pretrained model
        try:
            logger.info("Pretrained model not found locally. Downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, model_path)
            logger.info(f"Successfully downloaded pretrained model to {model_path}")
            return
        
        except Exception as download_error:
            logger.warning(
                f"Failed to download pretrained model: {download_error}"
            )
            logger.info("Falling back to training the model...")

        
        # Download failed → train the model
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e