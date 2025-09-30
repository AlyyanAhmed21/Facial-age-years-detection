from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_dir: Path
    dataset_config: str

@dataclass(frozen=True)
class DataPreparationConfig: # <<< NEW DATACLASS
    root_dir: Path
    raw_data_path: Path
    cleaned_data_path: Path

# ... (other configs are above)

@dataclass(frozen=True)
class MultiTaskModelTrainerConfig:
    root_dir: Path
    data_path: str
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