import cv2
from PIL import Image

from torchvision import transforms

from emotion_detection.exception import CustomException


class ModelProcessor:

    def __init__(self, config):
        self.config = config

        self.inference_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

    def process_face(self, face, device):

        try:
            image = Image.fromarray(
                cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            )

            image_tensor = self.inference_transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor.to(device)
        except Exception as e:
            raise CustomException(e)