from datasets import load_from_disk
import pandas as pd
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataPreparationConfig
from pathlib import Path
from PIL import Image # <<< ADD THIS IMPORT
import io              # <<< ADD THIS IMPORT

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def create_cleaned_dataframe(self):
        try:
            logger.info("Loading raw dataset to create cleaned CSV...")
            raw_dataset = load_from_disk(self.config.raw_data_path)
            
            df_train = raw_dataset['train'].to_pandas()
            df_val = raw_dataset['validation'].to_pandas()
            combined_df = pd.concat([df_train, df_val], ignore_index=True)
            
            image_dir = Path("artifacts/data_preparation/images")
            image_dir.mkdir(parents=True, exist_ok=True)
            
            combined_df['image_file_path'] = [
                str(image_dir / f"{i}.jpg") for i in range(len(combined_df))
            ]
            
            # --- IMPORTANT ---
            # We only need the file path for the CSV, so we drop the bulky 'image' column
            final_df_for_csv = combined_df.drop(columns=['image'])
            
            logger.info(f"Saving cleaned metadata to {self.config.cleaned_data_path}")
            final_df_for_csv.to_csv(self.config.cleaned_data_path, index=False)

            # --- CORRECTED IMAGE SAVING LOOP ---
            logger.info(f"Deterministically saving images to {image_dir}...")
            for i, row in combined_df.iterrows():
                image_path = Path(row['image_file_path'])
                image_dict = row['image']
                
                # Recreate the PIL Image from the dictionary's bytes data
                pil_image = Image.open(io.BytesIO(image_dict['bytes']))
                
                # Now save the reconstructed PIL Image
                pil_image.save(image_path)

        except Exception as e:
            logger.error(f"Failed during data preparation. Error: {e}")
            raise e