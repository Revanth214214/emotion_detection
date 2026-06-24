from emotion_detection import *
from emotion_detection.logging import logger
from emotion_detection.utils.common import read_yaml, create_directories
from emotion_detection.entity.config_entity import DatainjectionConfig
from emotion_detection.constant import *

class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_injection_config(self) -> DatainjectionConfig:
        config = self.config.data_injection

        create_directories([config.root_dir])

        data_injection_config = DatainjectionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_injection_config