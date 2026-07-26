from pathlib import Path

from emotion_detection.constant import *
from emotion_detection.utils.common import read_yaml, create_directories

from emotion_detection.entity.config_entity import (ModelLoaderConfig, ModelProcessorConfig)


class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_model_loader_config(self) -> ModelLoaderConfig:
        config = self.config.model_loader

        create_directories([config.root_dir])

        model_loader_config = ModelLoaderConfig(
            root_dir=Path(config.root_dir),
            trained_model=config.trained_model,
            local_model=Path(config.local_model),
            num_classes=config.num_classes
        )
        return model_loader_config

    def get_model_processor_config(self) -> ModelProcessorConfig:
        config = self.config.model_processor

        create_directories([config.root_dir])

        model_processor_config = ModelProcessorConfig(
            root_dir=Path(config.root_dir),
            image_size=tuple(config.image_size),
            mean=config.mean,
            std=config.std
        )

        return model_processor_config