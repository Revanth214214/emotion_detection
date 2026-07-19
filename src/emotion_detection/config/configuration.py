from emotion_detection import *
from emotion_detection.logging import logger
from emotion_detection.utils.common import read_yaml, create_directories
from emotion_detection.entity.config_entity import DataIngestionConfig
from emotion_detection.entity.config_entity import DataValidationConfig
from emotion_detection.entity.config_entity import DataTransformationConfig
from emotion_detection.entity.config_entity import ModelTrainerConfig
from emotion_detection.constant import *
import os

class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            ALL_REQUIRED_FILES = config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        # Automatically creates the artifacts/data_transformation root directory
        create_directories([config.root_dir])

        # Dynamically building the path: artifacts/data_transformation/Organized
        transformed_path = Path(os.path.join(config.root_dir, "Organized"))

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            organized_dir=Path(config.organized_dir),
            transformed_dir=transformed_path
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config_info = self.config['model_trainer']
        params_info = self.params['ModelTrainer']
        
        return ModelTrainerConfig(
            root_dir=Path(config_info['root_dir']),
            train_data_path=Path(config_info['train_data_path']),
            test_data_path=Path(config_info['test_data_path']),
            model_name=config_info['model_name'],
            batch_size=params_info['batch_size'],
            num_epochs=params_info['num_epochs'],
            patience=params_info['patience'],
            seed=params_info['seed'],
            lr_layer3=params_info['lr_layer3'],
            lr_layer4=params_info['lr_layer4'],
            lr_fc=params_info['lr_fc']
        )