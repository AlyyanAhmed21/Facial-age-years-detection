import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
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
from cnnClassifier.entity.config_entity import MultiTaskModelTrainerConfig
from cnnClassifier import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, model_name, num_labels_age, num_labels_gender, num_labels_race):
        super().__init__()
        self.efficientnet_base = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        original_classifier = self.efficientnet_base.classifier
        feature_dim = original_classifier.in_features
        self.efficientnet_base.classifier = nn.Identity()

        self.age_classifier = nn.Linear(feature_dim, num_labels_age)
        self.gender_classifier = nn.Linear(feature_dim, num_labels_gender)
        self.race_classifier = nn.Linear(feature_dim, num_labels_race)

    def forward(self, pixel_values, labels=None):
        features = self.efficientnet_base.efficientnet(pixel_values)
        pooled_features = features.last_hidden_state.mean(dim=[2, 3])
        age_logits = self.age_classifier(pooled_features)
        gender_logits = self.gender_classifier(pooled_features)
        race_logits = self.race_classifier(pooled_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            age_loss = loss_fct(age_logits, labels[:, 0])
            gender_loss = loss_fct(gender_logits, labels[:, 1])
            race_loss = loss_fct(race_logits, labels[:, 2])
            loss = (2.0 * age_loss) + gender_loss + race_loss

        return {"loss": loss, "age_logits": age_logits, "gender_logits": gender_logits, "race_logits": race_logits}

class FairFaceDataset(Dataset):
    def __init__(self, dataframe, processor, transforms):
        self.dataframe = dataframe
        self.processor = processor
        self.transforms = transforms
        self.normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_file_path']
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transforms(image)
        pixel_values = self.normalize(pixel_values)
        
        labels = torch.tensor([row['age_id'], row['gender_id'], row['race_id']], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

def compute_multitask_metrics(eval_pred):
    predictions, labels = eval_pred
    age_logits, gender_logits, race_logits = predictions['age_logits'], predictions['gender_logits'], predictions['race_logits']
    age_preds = np.argmax(age_logits, axis=1)
    gender_preds = np.argmax(gender_logits, axis=1)
    race_preds = np.argmax(race_logits, axis=1)
    age_labels, gender_labels, race_labels = labels[:, 0], labels[:, 1], labels[:, 2]
    age_acc = accuracy_score(age_labels, age_preds)
    gender_acc = accuracy_score(gender_labels, gender_preds)
    race_acc = accuracy_score(race_labels, race_preds)
    overall_acc = (age_acc + gender_acc + race_acc) / 3.0
    return {"age_accuracy": age_acc, "gender_accuracy": gender_acc, "race_accuracy": race_acc, "overall_accuracy": overall_acc}

class MultiTaskModelTrainer:
    def __init__(self, config: MultiTaskModelTrainerConfig):
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(config.model_name)

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        logger.info("Loading and preparing dataset from cleaned CSV...")
        df = pd.read_csv(self.config.data_path)
        
        label_maps = {}
        for task in ['age', 'gender', 'race']:
            labels = sorted(df[task].unique())
            label_maps[f'{task}_label2id'] = {label: i for i, label in enumerate(labels)}
            df[f'{task}_id'] = df[task].map(label_maps[f'{task}_label2id'])
        
        num_classes_age = len(label_maps['age_label2id'])
        num_classes_gender = len(label_maps['gender_label2id'])
        num_classes_race = len(label_maps['race_label2id'])
        train_df, test_df = train_test_split(df, test_size=self.config.test_split_size, random_state=self.config.random_state, stratify=df['age'])
        
        train_transforms = Compose([
            Resize((self.config.image_size, self.config.image_size)),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(), # Normalization is now in the Dataset
        ])
        
        val_transforms = Compose([
            Resize((self.config.image_size, self.config.image_size)),
            ToTensor(),
        ])
        
        train_dataset = FairFaceDataset(dataframe=train_df, processor=self.processor, transforms=train_transforms)
        test_dataset = FairFaceDataset(dataframe=test_df, processor=self.processor, transforms=val_transforms)

        model = MultiTaskEfficientNet(model_name=self.config.model_name, num_labels_age=num_classes_age, num_labels_gender=num_classes_gender, num_labels_race=num_classes_race).to(device)
        
        args = TrainingArguments(
            output_dir=self.config.root_dir,
            logging_dir=f'{self.config.root_dir}/logs',
            evaluation_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model="eval_overall_accuracy",
            dataloader_num_workers=4,
            lr_scheduler_type='cosine',
            report_to="none"
        )
        
        class EvalTrainer(Trainer):
            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                has_labels = "labels" in inputs
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.get("loss")
                predictions = {"age_logits": outputs["age_logits"], "gender_logits": outputs["gender_logits"], "race_logits": outputs["race_logits"]}
                return (loss, predictions, inputs["labels"] if has_labels else None)

        trainer = EvalTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_multitask_metrics)

        trainer.train()
        
        logger.info(f"Saving final model and processor to {self.config.trained_model_path}")
        trainer.save_model(self.config.trained_model_path)
        self.processor.save_pretrained(self.config.trained_model_path)