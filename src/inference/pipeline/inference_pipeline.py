import cv2
import torch
from inference.config.configuration import ConfigurationManager
from inference.components.model_loader import ModelLoader
from inference.components.model_processor import ModelProcessor
from inference.components.predictor import Predictor


class InferencePipeline:

    def __init__(self):

        config = ConfigurationManager()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model
        model_loader_config = config.get_model_loader_config()

        model_loader = ModelLoader(model_loader_config)

        self.model = model_loader.load_model(self.device)

        # Processor
        model_processor_config = config.get_model_processor_config()

        self.processor = ModelProcessor(model_processor_config)

        # Predictor
        self.predictor = Predictor(self.model)

        # Face Detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def run_frame(self, frame, min_confidence=0.5):
        """Detect faces in a BGR image array and classify each one.

        Returns a list of dicts. If the top probability is below
        `min_confidence` the emotion is reported as "Uncertain" and the
        model's best guess is kept in `top_emotion`.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        results = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            image_tensor = self.processor.process_face(face, self.device)
            top_emotion, confidence, probs = self.predictor.predict(image_tensor)

            results.append({
                "emotion": top_emotion if confidence >= min_confidence else "Uncertain",
                "top_emotion": top_emotion,
                "confidence": float(confidence),
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(self.predictor.class_names, probs)
                },
                "box": [int(x), int(y), int(w), int(h)]
            })

        return results

    def run(self, image_path, min_confidence=0.5):
        """Path-based wrapper kept for scripts and notebooks."""
        frame = cv2.imread(image_path)
        if frame is None:
            return {"error": "Failed to load image"}
        return self.run_frame(frame, min_confidence)
