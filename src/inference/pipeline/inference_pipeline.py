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

    def run(self):

        cap = cv2.VideoCapture(0)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:

                face = frame[y:y+h, x:x+w]

                image_tensor = self.processor.process_face(
                    face,
                    self.device
                )

                prediction, confidence, _ = self.predictor.predict(
                    image_tensor
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{prediction} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                "Emotion Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()