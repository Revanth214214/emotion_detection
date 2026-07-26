from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelLoaderConfig:
    root_dir: Path
    trained_model: str
    local_model: Path
    num_classes: int

@dataclass(frozen=True)
class ModelProcessorConfig:
    root_dir: Path
    image_size: tuple
    mean: list
    std: list