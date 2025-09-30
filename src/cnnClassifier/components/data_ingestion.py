# src/cnnClassifier/components/data_ingestion.py

from datasets import load_dataset
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        """
        Downloads and saves the FairFace dataset from the Hugging Face Hub.
        """
        try:
            logger.info(f"Downloading dataset '{self.config.dataset_name}' from Hugging Face Hub...")
            
            # load_dataset handles everything: download, verification, and caching
            # It returns a DatasetDict, typically with 'train' and 'validation' splits
            fairface_dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config,
                cache_dir=self.config.root_dir # Use our root_dir for caching
            )
            
            # Save the downloaded dataset to our specified artifacts directory
            # This makes it a persistent part of our DVC pipeline
            save_path = Path(self.config.local_data_dir)
            fairface_dataset.save_to_disk(save_path)
            
            logger.info(f"Dataset successfully downloaded and saved to {save_path}")

            # Optional: Log the structure of the downloaded dataset
            logger.info(f"Dataset splits: {list(fairface_dataset.keys())}")
            logger.info(f"Training set features: {fairface_dataset['train'].features}")

        except Exception as e:
            logger.error(f"Failed to download or save dataset. Error: {e}")
            raise e