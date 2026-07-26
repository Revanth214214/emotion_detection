import torch

from emotion_detection.exception import CustomException


class Predictor:

    def __init__(self, model):

        self.model = model

        self.class_names = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise"
        ]

    def predict(self, image_tensor):

        try:

            with torch.no_grad():

                outputs = self.model(image_tensor)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, dim=1)

            return (
                self.class_names[predicted.item()],
                confidence.item(),
                probabilities.squeeze().tolist()
            )

        except Exception as e:
            raise CustomException(e)