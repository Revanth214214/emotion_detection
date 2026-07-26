import os
import urllib.request

import torch
import torch.nn as nn
from torchvision import models

from emotion_detection.logging import logger
from emotion_detection.exception import CustomException


class ModelLoader:

    def __init__(self, config):
        self.config = config

    def download_model(self):

        try:
            if os.path.exists(self.config.local_model):
                logger.info("Model already exists. Skipping download.")
                return
            os.makedirs(self.config.root_dir, exist_ok=True)

            logger.info("Downloading trained model...")

            urllib.request.urlretrieve(
                self.config.trained_model,
                self.config.local_model
            )

            logger.info("Model downloaded successfully.")
        except Exception as e:
            raise CustomException(e)

    def build_model(self):

        try:
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(
                    model.fc.in_features,
                    self.config.num_classes
                )
            )
            return model

        except Exception as e:
            raise CustomException(e)

    def load_model(self, device):
        try:
            self.download_model()
            model = self.build_model()
            model.load_state_dict(
                torch.load(
                    self.config.local_model,
                    map_location=device
                )
            )

            model.to(device)
            model.eval()
            logger.info("Model loaded successfully.")
            return model

        except Exception as e:
            raise CustomException(e)