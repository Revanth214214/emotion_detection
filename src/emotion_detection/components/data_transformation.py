import os
import zipfile
from pathlib import Path
import cv2
from emotion_detection.logging import logger
from emotion_detection.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save_data(self):
        # 1. Verify our source data from ingestion exists
        if not os.path.exists(self.config.organized_dir):
            raise FileNotFoundError(f"Source organized directory missing at: {self.config.organized_dir}")
            
        train_path = os.path.join(self.config.organized_dir, "train")
        test_path = os.path.join(self.config.organized_dir, "test")
        
        splits = [("train", train_path), ("test", test_path)]
        
        # 2. Loop through train and test splits
        for split_name, src_split_path in splits:
            if not os.path.exists(src_split_path):
                raise FileNotFoundError(f"{split_name} directory missing inside ingestion artifacts!")
                
            categories = os.listdir(src_split_path)
            logger.info(f"Transforming {split_name} split. Found {len(categories)} emotion classes.")
            
            for category in categories:
                src_cat_path = os.path.join(src_split_path, category)
                if not os.path.isdir(src_cat_path):
                    continue
                    
                # 3. Create corresponding target subfolders (e.g., artifacts/data_transformation/Organized/train/happy)
                dest_cat_path = os.path.join(self.config.transformed_dir, split_name, category)
                os.makedirs(dest_cat_path, exist_ok=True)
                
                # 4. Process individual images
                images = os.listdir(src_cat_path)
                for img_name in images:
                    src_img_path = os.path.join(src_cat_path, img_name)
                    dest_img_path = os.path.join(dest_cat_path, img_name)
                    
                    try:
                        img = cv2.imread(src_img_path)
                        if img is None:
                            continue
                        
                        # Resize to standard network dimensions
                        img_resized = cv2.resize(img, (224, 224))
                        
                        # Ensure it is explicitly forced to 3 channels (equivalent to grayscale runtime layout)
                        if len(img_resized.shape) == 2:
                            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                            
                        # Save processed image to disk
                        cv2.imwrite(dest_img_path, img_resized)
                    except Exception as e:
                        logger.warning(f"Failed processing image file {img_name}: {str(e)}")
                        
            logger.info(f"Successfully processed and saved {split_name} split to transformation directory.")