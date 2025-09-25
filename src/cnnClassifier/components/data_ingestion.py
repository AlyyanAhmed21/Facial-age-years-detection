import os
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset from Kaggle.
        Make sure to have your kaggle.json file in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env variables.
        """
        try:
            logger.info(f"Downloading dataset from kaggle: {self.config.dataset_name}")
            os.system(f"kaggle datasets download {self.config.dataset_name} -p {os.path.dirname(self.config.local_data_file)}")
            # The downloaded file will be named 'facial-age.zip'. We need to rename it to 'data.zip' as per our config.
            downloaded_zip_path = os.path.join(os.path.dirname(self.config.local_data_file), 'facial-age.zip')
            os.rename(downloaded_zip_path, self.config.local_data_file)
            logger.info(f"Dataset downloaded and saved at {self.config.local_data_file}")
        except Exception as e:
            logger.error(f"Failed to download dataset. Error: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Dataset extracted to {unzip_path}")