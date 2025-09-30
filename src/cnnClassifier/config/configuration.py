from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataPreparationConfig,
    MultiTaskModelTrainerConfig # <-- Import the new one
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config, 
            local_data_dir=config.local_data_dir
        )
        return data_ingestion_config
    
    def get_data_preparation_config(self) -> DataPreparationConfig: # <<< NEW METHOD
        config = self.config.data_preparation

        create_directories([config.root_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir=config.root_dir,
            raw_data_path=config.raw_data_path,
            cleaned_data_path=config.cleaned_data_path
        )
        return data_preparation_config

    def get_multi_task_model_trainer_config(self) -> MultiTaskModelTrainerConfig:
        config = self.config.multi_task_model_trainer
        params = self.params
        create_directories([config.root_dir])

        multi_task_model_trainer_config = MultiTaskModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=config.data_path,
            trained_model_path=Path(config.trained_model_path),
            model_name=config.model_name,
            image_size=int(params.IMAGE_SIZE),
            learning_rate=float(params.LEARNING_RATE),
            batch_size=int(params.BATCH_SIZE),
            num_train_epochs=int(params.NUM_TRAIN_EPOCHS),
            weight_decay=float(params.WEIGHT_DECAY),
            warmup_steps=int(params.WARMUP_STEPS),
            test_split_size=float(params.TEST_SPLIT_SIZE),
            random_state=int(params.RANDOM_STATE)
        )
        return multi_task_model_trainer_config