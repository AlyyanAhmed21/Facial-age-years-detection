import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from functools import partial
from datasets import Dataset, Image, ClassLabel
from imblearn.over_sampling import RandomOverSampler
from transformers import (
    EfficientFormerImageProcessor, 
    EfficientFormerForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import (
    Compose, 
    Normalize, 
    RandomRotation, 
    RandomHorizontalFlip,
    Resize,
    ToTensor
)
import evaluate
from cnnClassifier.entity.config_entity import ModelTrainerConfig
from cnnClassifier import logger

# ==============================================================================
# TOP-LEVEL FUNCTION DEFINITIONS (FOR PICKLING)
# ==============================================================================

def apply_transforms(batch, processor, transform_pipeline):
    """Applies a given transformation pipeline to a batch of images."""
    # Create the normalization transform with stats from the processor
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    
    # Combine the base transforms with normalization
    full_transforms = Compose([*transform_pipeline.transforms, normalize])
    
    # Apply to each image in the batch
    batch["pixel_values"] = [full_transforms(img.convert("RGB")) for img in batch["image"]]
    return batch

def collate_fn(batch):
    """A custom collate function for image classification."""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def compute_metrics(eval_pred):
    """Computes accuracy metric for evaluation."""
    accuracy = evaluate.load("accuracy")
    predictions, label_ids = eval_pred
    predicted_labels = predictions.argmax(axis=1)
    return accuracy.compute(predictions=predicted_labels, references=label_ids)

# ==============================================================================

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _prepare_data(self):
        logger.info("Preparing data...")
        label_dict = {'001': '01', '002': '02', '003': '03', '004': '04', '005': '05', 
                      '006': '06-07', '007': '06-07', '008': '08-09', '009': '08-09', 
                      '010': '10-12', '011': '10-12', '012': '10-12', '013': '13-15', 
                      '014': '13-15', '015': '13-15', '016': '16-20', '017': '16-20', 
                      '018': '16-20', '019': '16-20', '020': '16-20', '021': '21-25', 
                      '022': '21-25', '023': '21-25', '024': '21-25', '025': '21-25', 
                      '026': '26-30', '027': '26-30', '028': '26-30', '029': '26-30', 
                      '030': '26-30', '031': '31-35', '032': '31-35', '033': '31-35', 
                      '034': '31-35', '035': '31-35', '036': '36-40', '037': '36-40', 
                      '038': '36-40', '039': '36-40', '040': '36-40', '041': '41-45', 
                      '042': '41-45', '043': '41-45', '044': '41-45', '045': '41-45', 
                      '046': '46-50', '047': '46-50', '048': '46-50', '049': '46-50', 
                      '050': '46-50', '051': '51-55', '052': '51-55', '053': '51-55', 
                      '054': '51-55', '055': '51-55', '056': '56-60', '057': '56-60', 
                      '058': '56-60', '059': '56-60', '060': '56-60', '061': '61-65', 
                      '062': '61-65', '063': '61-65', '064': '61-65', '065': '61-65', 
                      '066': '66-70', '067': '66-70', '068': '66-70', '069': '66-70', 
                      '070': '66-70', '071': '71-80', '072': '71-80', '073': '71-80', 
                      '074': '71-80', '075': '71-80', '076': '71-80', '077': '71-80', 
                      '078': '71-80', '079': '71-80', '080': '71-80', '081': '81-90', 
                      '082': '81-90', '083': '81-90', '084': '81-90', '085': '81-90', 
                      '086': '81-90', '087': '81-90', '088': '81-90', '089': '81-90', 
                      '090': '81-90', '091': '90+', '092': '90+', '093': '90+', 
                      '095': '90+', '096': '90+', '099': '90+', '100': '90+', 
                      '101': '90+', '110': '90+'}

        file_names, labels = [], []
        data_path = Path(self.config.data_path)
        for file in tqdm(sorted(data_path.glob('*/*.*'))):
            label = file.parent.name
            labels.append(label_dict[label])
            file_names.append(str(file))

        df = pd.DataFrame.from_dict({"image": file_names, "label": labels})
        ros = RandomOverSampler(random_state=self.config.random_state)
        df_resampled, y_resampled = ros.fit_resample(df[['image']], df['label'])
        df = pd.concat([df_resampled, y_resampled], axis=1)

        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        
        labels_list = sorted(list(set(labels)))
        label2id = {label: i for i, label in enumerate(labels_list)}
        id2label = {i: label for i, label in enumerate(labels_list)}
        
        ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
        dataset = dataset.map(lambda x: {'label': ClassLabels.str2int(x['label'])}, batched=True)
        dataset = dataset.cast_column('label', ClassLabels)
        
        split_dataset = dataset.train_test_split(test_size=self.config.test_split_size, shuffle=True, stratify_by_column="label")
        return split_dataset, id2label, label2id

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        split_dataset, id2label, label2id = self._prepare_data()
        train_data = split_dataset['train']
        test_data = split_dataset['test']
        
        processor = EfficientFormerImageProcessor.from_pretrained(self.config.model_name)
        
        # Define base transforms (without normalization)
        _train_transforms = Compose([
            Resize((self.config.image_size, self.config.image_size)),
            RandomRotation(15),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ])
        _val_transforms = Compose([
            Resize((self.config.image_size, self.config.image_size)),
            ToTensor(),
        ])

        # Use functools.partial to create specialized versions of our top-level function
        # This is a pickle-safe way to pass extra arguments (processor, transforms)
        train_transform_func = partial(apply_transforms, processor=processor, transform_pipeline=_train_transforms)
        val_transform_func = partial(apply_transforms, processor=processor, transform_pipeline=_val_transforms)

        train_data.set_transform(train_transform_func)
        test_data.set_transform(val_transform_func)
        
        model = EfficientFormerForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        ).to(device)

        args = TrainingArguments(
            output_dir=self.config.root_dir,
            logging_dir=f'{self.config.root_dir}/logs',
            evaluation_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=processor,
        )

        trainer.train()
        
        logger.info(f"Saving best model to {self.config.trained_model_path}")
        trainer.save_model(self.config.trained_model_path)