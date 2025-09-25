from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataPreparationConfig, ModelTrainerConfig

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
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        create_directories([config.root_dir])
        
        data_preparation_config = DataPreparationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            dataset_name=config.dataset_name
        )
        return data_preparation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            model_name=config.model_name,
            image_size=params.IMAGE_SIZE,
            learning_rate=params.LEARNING_RATE,
            batch_size=params.BATCH_SIZE,
            num_train_epochs=params.NUM_TRAIN_EPOCHS,
            weight_decay=params.WEIGHT_DECAY,
            warmup_steps=params.WARMUP_STEPS,
            test_split_size=params.TEST_SPLIT_SIZE,
            random_state=params.RANDOM_STATE
        )
        return model_trainer_config