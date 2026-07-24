import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
from emotion_detection.logging import logger
from emotion_detection.entity.config_entity import ModelTrainerConfig
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    def train(self):
        logger.info(f"Using device: {self.device}")

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

        train_dir = str(self.config.train_data_path)
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset   = datasets.ImageFolder(root=train_dir, transform=val_transform)

        logger.info(f"Total base train images found: {len(train_dataset)}")

        indices = list(range(len(train_dataset)))
        random.seed(42)
        random.shuffle(indices)

        split_size = int(self.config.val_split_size * len(indices))
        train_indices = indices[:split_size]
        val_indices   = indices[split_size:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset   = Subset(val_dataset, val_indices)

        logger.info(f"Train size: {len(train_subset)} | Val size: {len(val_subset)}")

        train_labels = [train_dataset.targets[i] for i in train_indices]
        class_counts = Counter(train_labels)
        
        total = len(train_labels)
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, sampler=sampler)
        val_loader   = DataLoader(val_subset,   batch_size=self.config.batch_size, shuffle=False)

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        for name, param in resnet.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(resnet.fc.in_features, 7)
        )
        
        model = resnet.to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = Adam([
            {"params": model.layer3.parameters(), "lr": self.config.lr_layer3},
            {"params": model.layer4.parameters(), "lr": self.config.lr_layer4},
            {"params": model.fc.parameters(),     "lr": self.config.lr_fc}
        ])

        best_val_acc     = 0.0
        patience_counter = 0

        logger.info("Starting model training pipeline stage...")

        for epoch in range(self.config.epochs):
            model.train()
            train_loss, train_correct = 0, 0

            for images, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}"
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_acc  = train_correct / len(train_subset) * 100
            train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss, val_correct = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                    val_loss    += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_acc  = val_correct / len(val_subset) * 100
            val_loss = val_loss / len(val_loader)

            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), str(self.config.model_path))
                logger.info(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{self.config.patience})")

                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        logger.info(f"Training Complete. Best validation accuracy achieved: {best_val_acc:.2f}%")