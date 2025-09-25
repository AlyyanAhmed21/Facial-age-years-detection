from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    data_path: Path
    dataset_name: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    model_name: str
    image_size: int
    learning_rate: float
    batch_size: int
    num_train_epochs: int
    weight_decay: float
    warmup_steps: int
    test_split_size: float
    random_state: int