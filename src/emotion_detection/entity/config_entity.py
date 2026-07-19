from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen = True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    organized_dir: Path
    transformed_dir: Path

@dataclass(frozen = True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    val_split_size: float
    batch_size: int
    epochs: int
    patience: int
    lr_layer3: float
    lr_layer4: float
    lr_fc: float