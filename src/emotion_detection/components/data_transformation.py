import os
import zipfile
from pathlib import Path
from emotion_detection.logging import logger
from emotion_detection.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def extract_data(self):
        """
        Extracts data.zip from the data_ingestion directory.
        """
        # Hardcoding the path to data.zip relative to the project root, just like the notebook
        zip_path = "artifacts/data_ingestion/data.zip"
        extract_path = "artifacts/data_ingestion"
        
        logger.info(f"Extracting raw data archive from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info("Extraction step complete.")

    def verify_and_prepare_data(self):
        """
        Validates the dataset structure and ensures train/test directories are populated.
        """
        if not os.path.exists(self.config.organized_dir):
            raise FileNotFoundError(f"Organized dataset directory missing at: {self.config.organized_dir}")
            
        train_path = os.path.join(self.config.organized_dir, "train")
        test_path = os.path.join(self.config.organized_dir, "test")
        
        logger.info(f"Checking dataset directories under: {self.config.organized_dir}")
        
        for split_name, split_path in [("Train", train_path), ("Test", test_path)]:
            if os.path.exists(split_path):
                categories = os.listdir(split_path)
                logger.info(f"{split_name} split verified. Found {len(categories)} classes: {categories}")
                
                total_images = sum([len(os.listdir(os.path.join(split_path, cat))) for cat in categories if os.path.isdir(os.path.join(split_path, cat))])
                logger.info(f"Total images in {split_name} split: {total_images}")
            else:
                raise FileNotFoundError(f"{split_name} directory is missing inside the organized path!")

        logger.info("Data Transformation Stage completed successfully!")