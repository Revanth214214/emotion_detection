import urllib.request as request
import zipfile
from emotion_detection.logging import logger
from emotion_detection.utils.common import *
from emotion_detection.entity.config_entity import DataValidationConfig
from pathlib import Path

class Datavalidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    # simple python code that validates all files

    def validate_all_files_exists(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "Organized"))

            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"validation status: {validation_status}")
                
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e