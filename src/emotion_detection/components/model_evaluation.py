import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from emotion_detection.entity.config_entity import ModelEvaluationConfig

#3rd component update

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        # 1. Image Transforms
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 2. Load Test Dataset
        test_dataset = datasets.ImageFolder(root=self.config.test_data_path, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 3. Reconstruct Model Architecture & Load Trained Weights
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(resnet.fc.in_features, len(test_dataset.classes))
        )
        model = resnet.to(self.device)
        model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        model.eval()

        # 4. Evaluation Loop
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = outputs.argmax(1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        # 5. Compute Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print("--- Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

        # 6. Save Metrics to JSON
        scores = {
            "test_accuracy": float(accuracy),
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1_score": float(f1)
        }
        
        with open(self.config.metric_file_name, "w") as f:
            json.dump(scores, f, indent=4)