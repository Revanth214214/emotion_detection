import urllib.request as request
import zipfile
from emotion_detection.logging import logger
from emotion_detection.utils.common import *
from emotion_detection.entity.config_entity import DataValidationConfig
from emotion_detection.entity.config_entity import ModelTrainerConfig
from pathlib import Path

import random
import numpy as np
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    #reproducibility
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _get_data_loaders(self):
        #image augmentation and normalizations
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #loading image datasets directly from transformation artifacts
        base_train = datasets.ImageFolder(root=str(self.config.train_data_path), transform=train_transform)
        base_val = datasets.ImageFolder(root=str(self.config.train_data_path), transform=val_transform)
        test_set = datasets.ImageFolder(root=str(self.config.test_data_path), transform=val_transform)

        #85/15 split
        indices = list(range(len(base_train)))
        random.shuffle(indices)
        split = int(0.85 * len(indices))
        
        train_subset = Subset(base_train, indices[:split])
        val_subset = Subset(base_val, indices[split:])

        #calculate inverse frequencey weights to manage class imbalnce
        train_labels = [base_train.targets[i] for i in indices[:split]]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        #production data loaders
        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, sampler=sampler)
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, len(train_subset), len(val_subset), test_set.classes

    def _build_model(self, num_classes=7):
        # Load the base network and target specific layers for training
        #load the base network and target specific layers for trainin
        model = models.resnet18(weights="IMAGENET1K_V1")
        for name, param in model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        #trail 2 dropout layer
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model.to(device)

    def initiate_model_training(self):
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        train_loader, val_loader, test_loader, train_size, val_size, classes = self._get_data_loaders()
        model = self._build_model(num_classes=len(classes))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam([
            {"params": model.layer3.parameters(), "lr": self.config.lr_layer3},
            {"params": model.layer4.parameters(), "lr": self.config.lr_layer4},
            {"params": model.fc.parameters(),     "lr": self.config.lr_fc}
        ])

        best_val_acc = 0.0
        patience_counter = 0
        save_path = os.path.join(self.config.root_dir, self.config.model_name)
            #training
        for epoch in range(self.config.num_epochs):
            model.train()
            train_loss, train_correct = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc = (train_correct / train_size) * 100
            avg_train_loss = train_loss / len(train_loader)

            #validation
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_acc = (val_correct / val_size) * 100
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{self.config.num_epochs}] Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"  Superior performance saved to disk ({best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement recorded ({patience_counter}/{self.config.patience})")
                if patience_counter >= self.config.patience:
                    print(f"Early stopping rule executed at epoch {epoch+1}")
                    break

        #out of sample evaluation
        print('/n final test evaluation')
        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        test_acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        print(f"Final Test Accuracy Score: {test_acc:.2f}%\n")
        print(classification_report(all_labels, all_preds, target_names=classes))